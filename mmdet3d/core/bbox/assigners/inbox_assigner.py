from matplotlib.pyplot import box
import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet3d.core.bbox.box_np_ops import points_in_rbbox


@BBOX_ASSIGNERS.register_module()
class InBoxAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(self, scale=4, pos_num=3):
        self.scale = scale
        self.pos_num = pos_num

    def assign(self, anchors, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to points.

        This method assign a gt bbox to every points set, each points set
        will be assigned with  the background_label (-1), or a label number.
        -1 is background, and semi-positive number is the index (0-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to the background_label (-1)
        2. A point is assigned to some gt bbox if
            (i) the point is within the k closest points to the gt bbox
            (ii) the distance between this point and the gt is smaller than
                other gt bboxes

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
                NOTE: currently unused.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        anchors_np = anchors.cpu().numpy()
        points = anchors_np[:, :3]
        points[:, 2] += anchors_np[:, 5] / 2
        gt_bboxes = gt_bboxes.cpu().numpy()
        num_points = points.shape[0]
        num_gts = gt_bboxes.shape[0]

        # import matplotlib.pyplot as plt
        # box_points = gt_bboxes[:, :3]
        # box_points[:, 2] += gt_bboxes[:, 5] / 2
        # points = points[:, :2]
        # box_points = box_points[:, :2]
        # plt.scatter(points[:, 0], points[:, 1], s=1)
        # plt.scatter(box_points[:, 0], box_points[:, 1])
        # plt.savefig("test.png", dpi=300)

        if num_gts == 0 or num_points == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points, ),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_full((num_points, ),
                                                  -1,
                                                  dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        assigned_gt_inds = anchors.new_zeros((num_points, ), dtype=torch.long)
        index = points_in_rbbox(points, gt_bboxes)
        for i in range(num_gts):
            idx = index[:, i].reshape(-1)
            assigned_gt_inds[idx] = i + 1

        # stores the assigned gt index of each point

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_points, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        # import matplotlib.pyplot as plt
        # idxs = assigned_gt_inds.cpu().numpy()
        # plt.scatter(points[:, 1], points[:, 2], s=0.1)
        # for i in range(num_gts):
        #     pts = points[idxs==i+1]
        #     plt.scatter(pts[:, 1], pts[:, 2], s=0.1)
        # plt.xlim((-10, 10))
        # plt.savefig("test.png", dpi=300)
        # plt.close()



        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
