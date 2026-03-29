from src.config import load_config
from train import train_from_config


def main():
    config = load_config("configs/resnet101_teacher_resnet50_student.json")
    train_from_config(config)


if __name__ == "__main__":
    main()
