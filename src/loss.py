from abc import ABC

import torch
from .criterion import Criterion, HungarianMatcher

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SSNLoss(Loss):
    def __init__(self, entity_types, device, model, optimizer, scheduler, max_grad_norm, discontinuous=False):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        matcher = HungarianMatcher(cost_class=2.0, cost_span=2.0)
        losses = ['class', 'boundary']
        weight_dict = {'loss_class': 2, 'loss_boundary': 2}
        self.criterion = Criterion(entity_types, matcher, weight_dict, losses)
        self.criterion.to(device)
        self._max_grad_norm = max_grad_norm
        self._discontinuous = discontinuous


    def compute(self, entity_logits, entity_bdy, entity_types, entity_spans_token, entity_masks):
        if self._discontinuous:
            return self.compute_discontinuous(entity_logits, entity_bdy, entity_types, entity_spans_token, entity_masks)

        outputs = {"pred_logits":entity_logits, "pred_boundaries":entity_bdy}

        entity_types = entity_types.masked_select(entity_masks)
        if len(entity_types) == 0:
            # print("----------No entity in the whole batch----------")
            return 0.
        sizes = [i.sum() for i in entity_masks]
        entity_masks = entity_masks.unsqueeze(2).repeat(1, 1, 2)
        spans = entity_spans_token.masked_select(entity_masks).view(-1, 2)
        
        targets = {"labels":entity_types, "spans":spans, "sizes":sizes}
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        train_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()

        return train_loss.item()


    def compute_discontinuous(self, entity_logits, entity_bdy, entity_types, entity_spans_token, entity_masks):
        # {flat, two, three}
        train_loss = 0.
        loss_weights = {'flat': 1.0, 'two': 0.5, 'three': 0.5}
        for key in entity_types.keys():
            entity_types_key, entity_masks_key, entity_spans_token_key = entity_types[key], entity_masks[key], entity_spans_token[key]
            outputs = {"pred_logits":entity_logits[key], "pred_boundaries":(entity_bdy[0][key], entity_bdy[1][key])}
            entity_types_key = entity_types_key.masked_select(entity_masks_key)
            if len(entity_types_key) == 0:
                # print("----------No entity in the whole batch----------")
                continue
            span_len = entity_spans_token_key.shape[-1]
            sizes = [i.sum() for i in entity_masks_key]
            entity_masks_key = entity_masks_key.unsqueeze(2).repeat(1, 1, span_len)
            spans = entity_spans_token_key.masked_select(entity_masks_key).view(-1, span_len)
            
            targets = {"labels":entity_types_key, "spans":spans, "sizes":sizes}
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            train_loss += sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) * loss_weights[key]
        
        if train_loss != 0.:
            # torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            # self._optimizer.step()
            # self._scheduler.step()
            # self._model.zero_grad()
            return train_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()

        return train_loss.item()
