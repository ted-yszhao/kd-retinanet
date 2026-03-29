import torch
import torch.nn as nn

from src.models.feature_extractor import extract_fpn_feats


class RetinaNetDistiller(nn.Module):
    '''
    
    '''
    def __init__(self, teacher, student, kd_loss, kd_weight=0.5):
        super().__init__()
        self.teacher = teacher.eval()
        self.student = student
        self.kd_loss = kd_loss
        self.kd_weight = kd_weight

        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, images, targets=None, branch="student"):
        if targets is None:
            if branch == "teacher":
                with torch.no_grad():
                    return self.teacher(images)
            if branch in {"student", "distiller", "full"}:
                return self.student(images)
            raise ValueError(f"Unsupported inference branch: {branch}")

        student_loss_dict = self.student(images, targets)
        det_loss = sum(student_loss_dict.values())

        with torch.no_grad():
            teacher_feats = extract_fpn_feats(self.teacher, images)

        student_feats = extract_fpn_feats(self.student, images)
        kd = self.kd_loss(student_feats, teacher_feats)

        total = det_loss + self.kd_weight * kd

        return {
            "loss_total": total,
            "loss_det": det_loss,
            "loss_kd": kd,
            **student_loss_dict,
        }
