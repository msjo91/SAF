import logging
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from .fedit import FedIT, ClientUpdate
logger = logging.getLogger('fedfl')

class FedSALoRA(FedIT):

    def __init__(self, config, device):
        super().__init__(config, device)
        from src.utils.config import get_method_params
        params = get_method_params(config)
        self.aggregation_mode = params.get('aggregation_mode', 'sa')
        self._client_local_params = {}
        logger.info(f'FedSA-LoRA initialized: aggregation_mode={self.aggregation_mode}, aggregate lora_A + classifier only, lora_B kept client-local')

    def local_train(self, model, train_loader, client_id, **kwargs):
        update = super().local_train(model, train_loader, client_id, **kwargs)
        local_b = {}
        for (name, param) in update.state_dict.items():
            if 'lora_B' in name or 'lora_b' in name:
                local_b[name] = param.clone()
        self._client_local_params[client_id] = local_b
        return update

    def aggregate(self, global_model, client_updates, round_idx=-1, **kwargs):
        if not client_updates:
            raise ValueError('No client updates provided')
        if round_idx >= 0:
            self._current_round = round_idx
        total_samples = sum((u.num_samples for u in client_updates))
        if total_samples == 0:
            raise ValueError('Total samples is zero')
        similarity_metrics = self.compute_cross_client_similarity(client_updates)
        self._last_similarity = similarity_metrics
        if similarity_metrics.num_client_pairs > 0:
            ordering = similarity_metrics.get_ordering()
            logger.info(f'[R{self._current_round}] [FedSA-LoRA Similarity] A={similarity_metrics.a_similarity_mean:.4f}, B={similarity_metrics.b_similarity_mean:.4f}, ({ordering}), delta(A-B)={similarity_metrics.a_similarity_mean - similarity_metrics.b_similarity_mean:.4f}')
        weights = [u.num_samples / total_samples for u in client_updates]
        global_state = global_model.state_dict()
        param_names = client_updates[0].state_dict.keys()
        aggregated_count = 0
        skipped_count = 0
        for param_name in param_names:
            if 'lora_B' in param_name or 'lora_b' in param_name:
                skipped_count += 1
                continue
            params = [u.state_dict[param_name] for u in client_updates]
            aggregated_param = torch.zeros_like(params[0])
            for (param, weight) in zip(params, weights):
                aggregated_param += weight * param
            global_state[param_name] = aggregated_param
            aggregated_count += 1
        global_model.load_state_dict(global_state, strict=False)
        logger.info(f'[R{self._current_round}] FedSA-LoRA aggregate: {aggregated_count} params aggregated (A+cls), {skipped_count} params skipped (B)')
        if hasattr(self, '_method_logger') and self._method_logger is not None:
            avg_metrics = {}
            if client_updates:
                all_keys = set()
                for u in client_updates:
                    if u.metrics:
                        all_keys.update(u.metrics.keys())
                for key in all_keys:
                    values = [u.metrics[key] for u in client_updates if u.metrics and key in u.metrics]
                    if values:
                        avg_metrics[f'avg_{key}'] = sum(values) / len(values)
            avg_metrics.update(similarity_metrics.to_dict())
            self._method_logger.log_aggregation(round_num=self._current_round, num_clients=len(client_updates), total_samples=total_samples, aggregation_method='fedsalora', custom_metrics=avg_metrics)
            lr_factor = kwargs.get('lr_factor', 1.0)
            self._method_logger.log_aggregate_analysis(round_num=self._current_round, global_model=global_model, client_updates=client_updates, lr_factor=lr_factor, effective_lrs={'lr_a': self.lr_a * lr_factor, 'lr_b': self.lr_b * lr_factor, 'lr_cls': self.lr_cls * lr_factor}, similarity_metrics=similarity_metrics, has_magnitude=False)
        return global_model

    def get_client_state(self, client_id):
        return self._client_local_params.get(client_id)

    def get_participating_clients(self):
        return list(self._client_local_params.keys())
