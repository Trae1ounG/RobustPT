import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer_prompt_strategy as vit
import copy
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
import torch.nn.functional as F
def _get_clones(layer,num):
    return nn.ModuleList([copy.deepcopy(layer)  for i in range(num)])

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
        self.config = config
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)
        
        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["finetune_first"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if config["max_text_len"] != 40:
                state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
                pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
                state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.hatememes_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.food101_classifier.apply(objectives.init_weights)               
            
        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.mmimdb_classifier.apply(objectives.init_weights)  
            
        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)            
            print("use pre-finetune model")

        self.prompt_type = self.hparams.config["prompt_type"]
        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.embed_dim = embed_dim
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_prompt = self.hparams.config["multi_layer_prompt"]
        self.num_layers = self.hparams.config["num_layers"]
        self.TEXT_STEP = 0
        self.IMG_STEP = 1
        self.CONSENSUS_VIEW = [2,3,4]
        self.step_cnt = 0
        
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1
        from timm.models.layers import trunc_normal_
        
        print("prompt Num",prompt_num)
        print("prompt Length",prompt_length)
        
        #Multimodal Prompt Design & f_missing
        complete_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_img_prompt[:,1:2,:].fill_(1)            
        self.complete_img_prompt = nn.Parameter(complete_img_prompt)
        complete_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_text_prompt[:,2:3,:].fill_(1)            
        self.complete_text_prompt = nn.Parameter(complete_text_prompt)
        f_missing_img2text = heads.ResidualBlock(embed_dim)
        f_missing_text2img = heads.ResidualBlock(embed_dim)
        self.f_missing_img2text =  _get_clones(f_missing_img2text, prompt_num)
        self.f_missing_text2img =  _get_clones(f_missing_text2img, prompt_num)

        #Frozon Backbone        
        for param in self.transformer.parameters():
            param.requires_grad=False
        for param in self.text_embeddings.parameters():
            param.requires_grad=False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()
        
        # ===================== load downstream (test_only) ======================
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        self.records = {}

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None,
        step_num=0
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"] 
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"] 
        text_embeds = self.text_embeddings(text_ids) 
        
        img = batch[imgkey][0]     
        if image_embeds is None and image_masks is None:
                   
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
            
        else:
            patch_index, image_labels = (
                None,
                None,
            )
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        if step_num == self.TEXT_STEP:
            self.complete_img_prompt.requires_grad_(False)
            self.complete_text_prompt.requires_grad_(True)
        elif step_num == self.IMG_STEP:
            self.complete_img_prompt.requires_grad_(True)
            self.complete_text_prompt.requires_grad_(False)
        elif step_num in self.FINAL_STAGE:
            self.complete_img_prompt.requires_grad_(True)
            self.complete_text_prompt.requires_grad_(True)
        text_prompts = None
        img_prompts = None
        for idx in range(len(img)):
            tmp_text_prompts = []
            tmp_img_prompts = []
            for i in range(len(self.prompt_layers)):
                complete_img_prompt = self.complete_img_prompt[i,:,:]
                complete_text_prompt = self.complete_text_prompt[i,:,:]
                if batch["missing_type"][idx] == 1:
                    img_prompt = complete_img_prompt
                    text_prompt = self.f_missing_img2text[i](img_prompt)
                elif batch["missing_type"][idx] == 0:
                    text_prompt = complete_text_prompt
                    img_prompt = complete_img_prompt
                elif batch["missing_type"][idx] == 2:
                    text_prompt = complete_text_prompt
                    img_prompt = self.f_missing_text2img[i](text_prompt)
                tmp_text_prompts.append(text_prompt.unsqueeze(0))
                tmp_img_prompts.append(img_prompt.unsqueeze(0))
            text_prompt = torch.cat(tmp_text_prompts, dim=0)
            img_prompt = torch.cat(tmp_img_prompts, dim=0)
            if text_prompt.size(0) != 1:
                text_prompt = text_prompt.unsqueeze(0)
            if img_prompt.size(0) != 1:
                img_prompt = img_prompt.unsqueeze(0)
            if text_prompts is None:
                text_prompts = text_prompt
            else:
                text_prompts = torch.cat([text_prompts,text_prompt], dim=0)
            if img_prompts is None:
                img_prompts = img_prompt
            else:
                img_prompts = torch.cat([img_prompts,img_prompt], dim=0)
            
            
        if self.learnt_p:
            if self.prompt_type == 'head':
                prompt_masks = torch.ones(len(img), self.prompt_length*len(self.prompt_layers), dtype=text_prompts.dtype, device=text_prompts.device).long()
            elif self.prompt_type == 'cross':
                prompt_masks = torch.ones(len(img), self.prompt_length*len(self.prompt_layers), dtype=text_prompts.dtype, device=text_prompts.device).long()
                
        else:
            prompt_masks = torch.ones(len(img), self.prompt_length, dtype=text_prompts.dtype, device=text_prompts.device).long()   

        load_text_prompts, load_img_prompts = [], []
        
        if self.learnt_p:
            if self.prompt_type == 'cross':
                co_masks = torch.cat([prompt_masks,text_masks,prompt_masks,image_masks], dim=1)
            elif self.prompt_type == 'head':
                co_masks = torch.cat([prompt_masks,text_masks,image_masks], dim=1)
        else:
            co_masks = torch.cat([text_masks,image_masks], dim=1)
        x = (text_embeds,image_embeds)
        
        for i, blk in enumerate(self.transformer.blocks):
            #前i层加入提示
            if i in self.prompt_layers:
                idx = self.prompt_layers.index(i)               
                text_prompt = text_prompts[:,idx]
                img_prompt = img_prompts[:,idx] 
                load_text_prompts.append(text_prompt)
                load_img_prompts.append(img_prompt)
                text_embeds,img_embeds = blk(x, mask=co_masks, 
                                prompts=(text_prompt,img_prompt), 
                                learnt_p=self.learnt_p,
                                prompt_type=self.prompt_type)
                x = (text_embeds,img_embeds)
            else:
                text_embeds,img_embeds = blk(x, mask=co_masks,
                                             prompts=None,
                                             learnt_p=self.learnt_p,
                                            prompt_type=self.prompt_type)
                x = (text_embeds,img_embeds)
        x = torch.cat(list(x),dim=1)
        x = self.transformer.norm(x)
        text_embs_len, text_prompt_len, img_embs_len = text_embeds.shape[1], text_prompt.shape[1], img_embeds.shape[1]
        if self.learnt_p:
            total_prompt_len = len(self.prompt_layers)* text_prompts.shape[-2]
        else:
            total_prompt_len = 0            
        load_text_prompts = torch.cat(load_text_prompts,dim = 0)
        load_img_prompts = torch.cat(load_img_prompts,dim = 0)
        text_feats, image_feats = (
            x[:,:text_embs_len + total_prompt_len],
            x[:, -img_embs_len - total_prompt_len: ],
        )
        cls_feats = self.pooler(x[:, total_prompt_len:total_prompt_len+1])   
        raw_cls_feats = self.pooler(x[:, total_prompt_len:total_prompt_len+1])  
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": raw_cls_feats,
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
            "text_prompts":load_text_prompts,
            "image_prompts":load_img_prompts,
        }
        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        
        #KL Loss
        if self.step_cnt == self.TEXT_STEP:
            ret.update(objectives.compute_distance(self,batch,True))
        elif self.step_cnt == self.IMG_STEP:
            ret.update(objectives.compute_distance(self,batch,False))
            
        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch,self.step_cnt))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch,self.step_cnt))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch,self.step_cnt))
            
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch,self.step_cnt))
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch,self.step_cnt))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch,self.step_cnt))          
                
        self.step_cnt += 1
        self.step_cnt %= 5
        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        self.sample = True
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
