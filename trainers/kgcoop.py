import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    #"EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    #"EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.KGCOOP.N_CTX
        ctx_init = cfg.TRAINER.KGCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.KGCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        #print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()
        
        #prompts_ = [prompt_prefix + " " + name + "." for name in classnames]        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            self.text_features_nonorm = text_features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS


        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.KGCOOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.test_time_fuse = cfg.TRAINER.KGCOOP.TEST_TIME_FUSE
        self.test_time_weight = cfg.TRAINER.KGCOOP.TEST_TIME_WEIGHT
        if self.test_time_fuse:
            self.zs_weight = self.prompt_learner.text_features_nonorm

    def compute_info_gain(self, logits0, logits1):
        # logits0, logits1: n*c
        C = logits0.shape[1]
        H = math.log(C)
        probs0 = F.softmax(logits0, dim=-1) + 1e-5
        probs1 = F.softmax(logits1, dim=-1) + 1e-5
        H0 = -(probs0 * torch.log(probs0)).sum(dim=-1, keepdim=True)
        H1 = -(probs1 * torch.log(probs1)).sum(dim=-1, keepdim=True)
        return (H0 - H1) / H, H0 / H, H1 / H  # relative info gain

    def tune_text_features(self, text_features, image_features, logit_scale):
        zs_weight = self.zs_weight.to(text_features.device).type(text_features.dtype)

        delta_weight = text_features - zs_weight
        if self.test_time_fuse == "weight":
            text_features2 = zs_weight + delta_weight * self.test_time_weight
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ text_features2.t()
        elif self.test_time_fuse == "alpha":  # info gain
            zs_weight_ = zs_weight / zs_weight.norm(dim=-1, keepdim=True)
            text_features_ = text_features / text_features.norm(dim=-1, keepdim=True)
            logits0 = logit_scale * image_features @ zs_weight_.t()
            logits1 = logit_scale * image_features @ text_features_.t()
            infogain, H0, H1 = self.compute_info_gain(logits0.float(), logits1.float())
            infogain = infogain.type(text_features.dtype)
            weight = torch.sigmoid(infogain * self.test_time_weight)  # N*1
            text_features2 = zs_weight.unsqueeze(0) + weight[..., None] * delta_weight.unsqueeze(0)  # N*c*d
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features.unsqueeze(1) * text_features2).sum(dim=-1)
        elif self.test_time_fuse == "SOFTmsp":  # comparing msp of two classifiers
            zs_weight_ = zs_weight / zs_weight.norm(dim=-1, keepdim=True)  # c*d
            text_features_ = text_features / text_features.norm(dim=-1, keepdim=True)  # c*d
            logits0 = logit_scale * image_features @ zs_weight_.t()  # b*c
            logits1 = logit_scale * image_features @ text_features_.t()  # b*c
            probs0 = F.softmax(logits0, dim=-1)
            probs1 = F.softmax(logits1, dim=-1)
            max_prob0 = torch.max(probs0, dim=1, keepdim=True)[0].unsqueeze(-1)
            max_prob1 = torch.max(probs1, dim=1, keepdim=True)[0].unsqueeze(-1)
            weight = torch.sigmoid((max_prob1 - max_prob0) * self.test_time_weight)
            text_features2 = zs_weight.unsqueeze(0) * (1 - weight) + text_features.unsqueeze(0) * weight
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features.unsqueeze(1) * text_features2).sum(dim=-1)
        elif self.test_time_fuse == "SOFTenergy":  # comparing energy
            temp = 1 / logit_scale
            zs_weight_ = zs_weight / zs_weight.norm(dim=-1, keepdim=True)  # c*d
            text_features_ = text_features / text_features.norm(dim=-1, keepdim=True)  # c*d
            logits0 = logit_scale * image_features @ zs_weight_.t()  # b*c
            logits1 = logit_scale * image_features @ text_features_.t()  # b*c
            num_classes = logits0.shape[1]
            beta = 1.0 / (math.log(num_classes) + 1 / temp)
            neg_energy0 = beta * temp * torch.logsumexp(logits0, dim=-1, keepdim=True).unsqueeze(-1)
            neg_energy1 = beta * temp * torch.logsumexp(logits1, dim=-1, keepdim=True).unsqueeze(-1)
            weight = torch.sigmoid((neg_energy1 - neg_energy0) * self.test_time_weight)
            text_features2 = zs_weight.unsqueeze(0) * (1 - weight) + text_features.unsqueeze(0) * weight
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features.unsqueeze(1) * text_features2).sum(dim=-1)

        return logits

    def forward(self, image):
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features_old = self.ori_embedding

        if self.test_time_fuse:
            logits = self.tune_text_features(text_features, image_features, logit_scale)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        score = cos(text_features, text_features_old)
        score = 1.0 - torch.mean(score)

        return logits, score


@TRAINER_REGISTRY.register()
class KgCoOp(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.KGCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.KGCOOP.PREC == "fp32" or cfg.TRAINER.KGCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.w = cfg.TRAINER.KGCOOP.W

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            #if "prompt_learner" not in name: # and "adapter" not in name:
            if "ctx" not in name: 
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        #self.optim_ = build_optimizer(self.model.adapter, cfg.OPTIM)
        #self.sched_ = build_lr_scheduler(self.optim, cfg.OPTIM)
        #self.register_model('clip_adapter', self.model.adapter, self.optim_, self.sched_)

        self.scaler = GradScaler() if cfg.TRAINER.KGCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.KGCOOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output,score = self.model(image)
            loss = F.cross_entropy(output, label)+self.w*score
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            #self.update_lr()
            self.sched.step()
            #self.sched_.step()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def model_inference(self, input):
        return self.model(input)[0]


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
