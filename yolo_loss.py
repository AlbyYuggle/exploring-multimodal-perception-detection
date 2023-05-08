import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        ret_matrix = torch.zeros(boxes.shape).to(boxes.device)
        ret_matrix[:, 0] = boxes[:, 0] / self.S - 0.5 * boxes[:, 2]
        ret_matrix[:, 1] = boxes[:, 1] / self.S - 0.5 * boxes[:, 3]
        ret_matrix[:, 2] = boxes[:, 0] / self.S + 0.5 * boxes[:, 2]
        ret_matrix[:, 3] = boxes[:, 1] / self.S + 0.5 * boxes[:, 3]
        boxes = ret_matrix

        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 4) ...]
        box_target : (tensor)  size (-1, 5)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the self.B bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        best_ious = torch.zeros(box_target.shape[0], 1).to(box_target.device)
        best_boxes = torch.zeros(box_target.shape[0], 5).to(box_target.device)
        # print("CALLED BESTIOU")

        # print(len(pred_box_list), pred_box_list[0].shape, box_target.shape)

        potential_boxes = []
        potential_ious = []
        for b in range(self.B):
            preds = pred_box_list[b][:, :4]
            preds = self.xywh2xyxy(preds)
            # print(preds.shape, box_target[:, :4].shape)
            box_targets = self.xywh2xyxy(box_target[:, :4])
            # print(preds.shape, box_target.shape)
            ious = compute_iou(preds, box_targets)
            # print(ious.shape, best_ious.shape, preds.shape, best_boxes.shape)
            cur_best_boxes = torch.argmax(ious, dim=0)
            cur_best_ious = torch.max(ious, dim=0)[0]

            potential_boxes.append(cur_best_boxes)
            potential_ious.append(cur_best_ious)
            # # print(ious.shape, cur_best_boxes.shape, cur_best_ious.shape, best_ious.shape, best_boxes.shape, preds.shape, 'aa')
            # # print((cur_best_ious > torch.squeeze(best_ious)).shape, 'a')
            # tbc = cur_best_ious >= torch.squeeze(best_ious)
            
            # # print(tbc.shape, 'b', tbc)
            # best_boxes[tbc, :4] = preds[cur_best_boxes[tbc]]
            # best_boxes[tbc, 4] = pred_box_list[b][cur_best_boxes[tbc], 4]

            # best_ious[tbc, 0] = cur_best_ious[tbc]

        potential_boxes = torch.stack(potential_boxes, dim=0)
        potential_ious = torch.stack(potential_ious, dim=0)
        # print(potential_boxes.shape, potential_ious.shape)
        indices = torch.argmax(potential_ious, dim=0)
        best_ious = torch.max(potential_ious, dim=0)[0]
        stacked_box_list = torch.stack(pred_box_list, dim=0)
        # print(stacked_box_list.shape)
        best_boxes = stacked_box_list[indices, torch.arange(stacked_box_list.shape[1]), :]
        # print(best_boxes.shape, stacked_box_list.shape, indices.shape, best_ious.shape)
        
        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        #print(classes_pred.shape, classes_target.shape, has_object_map.shape)
        class_loss = torch.sum(has_object_map*torch.sum((classes_pred - classes_target) ** 2, axis=-1))
        return class_loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        no_obj_loss = 0
        # stacked = torch.stack(pred_boxes_list, dim=0)
        # print(stacked.shape, has_object_map.shape, 'a')
        # stacked_obj = has_object_map.unsqueeze(0).expand_as(stacked[:, :, :, :, 4])
        # no_obj_loss = torch.sum((stacked[:, :, :, :, 4] * (1 - stacked_obj.float()))**2)
        for box in pred_boxes_list:
            no_obj_loss += torch.sum((box[:, :, :, 4] * (1 - has_object_map.float()))**2)
        return no_obj_loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        contain_loss = torch.sum((box_pred_conf - box_target_conf.detach()) ** 2)
        return contain_loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        reg_loss = torch.sum((box_pred_response[:, 0:2] - box_target_response[:, 0:2]) ** 2) + torch.sum((torch.sqrt(box_pred_response[:, 2:4]) - torch.sqrt(box_target_response[:, 2:4])) ** 2) 
        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        total_loss = 0.0

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        pred_boxes_list = []
        pred_cls = pred_tensor[:, :, :, -3:]
        for i in range(self.B):
            pred_boxes_list.append(pred_tensor[:, :, :, i * 5: (i + 1) * 5])

        # compute classification loss
        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)

        # compute no-object loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation

        # print("reshaping boxes")
        # print(pred_boxes_list[0].shape)
        has_object_map = has_object_map.view(-1, 1)
        pred_boxes_list = [pred_boxes_list[i].reshape(-1, 5)[torch.squeeze(has_object_map), :] for i in range(self.B)]
        target_boxes = target_boxes.view(-1, 4)[torch.squeeze(has_object_map), :]
        
        # print(pred_boxes_list[0].shape, len(pred_boxes_list))

        # print("finding best iou boxes")
        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes)
        # print(best_boxes.shape, best_ious.shape)

        # print("computing regression loss")
        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        reg_loss = self.get_regression_loss(best_boxes, target_boxes)

        # print("computing contain_object_loss")
        # compute contain_object_loss
        contain_obj_loss = self.get_contain_conf_loss(best_boxes[:, 4], best_ious)
        
        # print("computing final loss")
        # compute final loss
        total_loss = cls_loss + self.l_coord * reg_loss + self.l_noobj * no_obj_loss + contain_obj_loss

        # construct return loss_dict
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=reg_loss,
            containing_obj_loss=contain_obj_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=cls_loss,
        )
        return loss_dict