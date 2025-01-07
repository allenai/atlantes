"""Module to do Active learning  and sampling for machine annotation based on model predictions.


WIP: This module is a work in progress and is not yet ready for use."""

# def get_examples_where_fishing_was_predicted_for_non_fishing_vessel(
#     avg_fishing_probs_per_track: torch.Tensor,
#     known_no_fishing_mask: torch.Tensor,
#     num_samples_to_ask_for: int,
# ) -> torch.Tensor:
#     """Get examples where fishing was predicted for a non-fishing vessel."""
#     # Get the indices where the fishing was predicted
#     topk, topk_indices = torch.topk(avg_fishing_probs_per_track, num_samples_to_ask_for)
#     predicted_fishing_mask = torch.where(avg_fishing_probs_per_track <= 1, 1.0, 0.0)
#     # Get the indices where the vessel is known to not be fishing
#     fishing_preds_on_non_fishing_vessels = torch.logical_and(
#         known_no_fishing_mask, predicted_fishing_mask
#     )
#     # Return True indices
#     fishing_preds_on_non_fishing_vessels_idxs = torch.nonzero(
#         fishing_preds_on_non_fishing_vessels
#     )
#     # Return intersection of the topk indices and the fishing preds on non fishing vessels
#     fishing_preds_on_non_fishing_vessels_idxs, counts = torch.cat(
#         [topk_indices, fishing_preds_on_non_fishing_vessels_idxs]
#     ).unique(return_counts=True)
#     sampled_idxs = fishing_preds_on_non_fishing_vessels_idxs[torch.where(counts.gt(1))]
#     if sampled_idxs.size()[0] < num_samples_to_ask_for:
#         logger.warning(f"Only {sampled_idxs.size()[0]} samples were found")
#     return sampled_idxs


# def sample_high_confidence_fishing_examples_from_known_non_fishing_vessels(
#     known_no_fishing_vessel_paths: list[str],
#     atlas_activity_model: AtlasActivity,
#     batch_size: int,
#     num_samples_known_non_fishing_vessel: int,
# ) -> list[str]:
#     """Sample hard examples from a list of ais data files.

#     THe paths do not include the bucket name or top level directory.


#     # TODO: Remove really short trajectories from the data
#     # Add component to stratify the inital samaple being passed ins o weessentially do this trategy over areas of interest
#     """
#     paths_array = np.array(known_no_fishing_vessel_paths)
#     data_stream = init_lazy_online_dataloader(
#         cpe_kernel_size=atlas_activity_model.cpe_kernel_size,
#         online_file_paths=known_no_fishing_vessel_paths,
#         batch_size=batch_size,
#     )

#     atlas_activity_model.to(DEVICE)

#     uncertainty_scores_predicted_fishing_batches = []
#     known_non_fishing_mask_batches = []
#     for i, x in enumerate(data_stream):
#         logger.info(f"Batch {i}")
#         subpath_class_outputs, attn_mask, binned_ship_types = model_forward_pass(
#             x, atlas_activity_model, DEVICE
#         )
#         # Outputs are shape (batch_size*subpath_dim, n_classes)
#         subpath_class_outputs = subpath_class_outputs.detach().cpu()

#         max_num_subpaths = subpath_class_outputs.shape[0] // batch_size
#         logger.info(f"num_subpaths{max_num_subpaths}")
#         attn_mask = attn_mask.detach().cpu()
#         class_probs = F.softmax(subpath_class_outputs, dim=1)

#         uncertainty_scores_predicted_fishing_batches.append(
#             get_mean_per_trajectory_of_fishing_probabilities(
#                 class_probs, attn_mask, max_num_subpaths
#             )
#         )
#         known_non_fishing_mask_batches.append(
#             torch.logical_and(binned_ship_types == 1, attn_mask)
#         )

#     uncertainty_scores_predicted_fishing_tensor = torch.cat(
#         [
#             uncertainty_scores_predicted_fishing.unsqueeze(0)
#             for uncertainty_scores_predicted_fishing in uncertainty_scores_predicted_fishing_batches
#         ]
#     )
#     known_non_fishing_mask_tensor = torch.cat(
#         [
#             known_non_fishing_mask.unsqueeze(0)
#             for known_non_fishing_mask in known_non_fishing_mask_batches
#         ]
#     )
#     ## Get Examples where fishing was predicted for a non-fishing vessel
#     fishing_preds_on_non_fishing_vessels_idxs = (
#         get_examples_where_fishing_was_predicted_for_non_fishing_vessel(
#             uncertainty_scores_predicted_fishing_tensor,
#             known_non_fishing_mask_tensor,
#             num_samples_known_non_fishing_vessel,
#         )
#     )

#     sampled_paths = paths_array[fishing_preds_on_non_fishing_vessels_idxs]
#     return sampled_paths.tolist()
