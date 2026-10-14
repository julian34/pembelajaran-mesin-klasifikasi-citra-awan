"""Pabrik model baseline untuk klasifikasi citra awan (7 kelas).

Transfer learning dengan bobot pra-latih ImageNet (torchvision). Setiap model
dimuat dengan bobot ImageNet lalu lapisan classifier terakhirnya diganti menjadi
`Linear(in_features, num_classes)`.

Arsitektur yang didukung:
  - "densenet121"  classifier: model.classifier            (in_features 1024) — torchvision, bobot ImageNet ✅
  - "alexnet"      classifier: model.classifier[6]         (in_features 4096) — torchvision, bobot ImageNet ✅
  - "densenet201"  classifier: model.classifier            (in_features 1920) — torchvision, bobot ImageNet ✅ (fallback 264)
  - "resnet18"     classifier: model.fc                    (in_features 512)  — torchvision, bobot ImageNet ✅
  - "resnet50"     classifier: model.fc                    (in_features 2048) — torchvision, bobot ImageNet ✅
  - "densenet264"  classifier: model.classifier            — torchvision TIDAK punya bobot ImageNet;
                   pretrained=True dibangun via timm ("densenet264d"), pretrained=False via torchvision (dari nol)

Catatan normalisasi: karena backbone dilatih di ImageNet, pakai mean/std ImageNet
(IMAGENET_MEAN/IMAGENET_STD di bawah), bukan statistik dataset. Lihat
docs/ARSITEKTUR_BASELINE.md Bagian 3.
"""
from __future__ import annotations

import torch.nn as nn

# Normalisasi preprocessing untuk backbone pra-latih ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Arsitektur yang punya bobot ImageNet di torchvision (transfer learning langsung jalan).
SUPPORTED_PRETRAINED = ("densenet121", "alexnet", "densenet201", "resnet18", "resnet50")
# densenet264 pra-latih tidak lewat torchvision melainkan timm (lihat build_model).
TIMM_PRETRAINED = {"densenet264": "densenet264d"}
FINETUNE_MODES = ("feature_extraction", "full")


def _freeze_backbone(model: nn.Module) -> None:
    """Bekukan semua parameter (dipanggil SEBELUM classifier baru dipasang)."""
    for param in model.parameters():
        param.requires_grad = False


def build_model(
    architecture: str,
    num_classes: int = 7,
    pretrained: bool = True,
    finetune: str = "feature_extraction",
) -> nn.Module:
    """Bangun model baseline dengan classifier diganti ke `num_classes`.

    Args:
        architecture: salah satu "densenet121", "alexnet", "densenet201",
            "densenet264".
        num_classes: jumlah kelas keluaran (default 7 untuk skema GCD).
        pretrained: True memuat bobot ImageNet; False melatih dari nol.
        finetune: "feature_extraction" membekukan backbone dan hanya melatih
            classifier baru; "full" melatih seluruh parameter.

    Returns:
        nn.Module dengan lapisan classifier terakhir berukuran `num_classes`.

    Raises:
        ValueError: arsitektur/mode tidak dikenal, atau kombinasi
            pretrained yang tidak tersedia.
    """
    arch = architecture.lower().strip()
    if finetune not in FINETUNE_MODES:
        raise ValueError(
            f"finetune={finetune!r} tidak dikenal; pilih salah satu {FINETUNE_MODES}."
        )

    # densenet264 pra-latih tidak ada di torchvision -> dibangun via timm.
    if arch == "densenet264" and pretrained:
        return _build_densenet264_timm(num_classes, finetune)

    # torchvision diimpor di dalam fungsi agar modul ini bisa diimpor untuk
    # inspeksi (mis. daftar arsitektur) tanpa torch terpasang.
    from torchvision import models

    if arch == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
    elif arch == "densenet201":
        weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet201(weights=weights)
    elif arch == "alexnet":
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.alexnet(weights=weights)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    elif arch == "densenet264":
        # Hanya sampai di sini bila pretrained=False (dari nol via torchvision).
        model = models.DenseNet(
            growth_rate=32,
            block_config=(6, 12, 64, 48),
            num_init_features=64,
            num_classes=num_classes,
        )
        return model  # classifier sudah berukuran num_classes sejak dibuat.
    else:
        raise ValueError(
            f"architecture={architecture!r} tidak dikenal. Pilihan: "
            "densenet121, densenet201, densenet264, alexnet, resnet18, resnet50."
        )

    # Bekukan backbone SEBELUM memasang classifier baru, agar parameter head
    # baru tetap trainable (requires_grad=True secara default).
    if finetune == "feature_extraction":
        _freeze_backbone(model)

    _replace_classifier(model, arch, num_classes)
    return model


def _replace_classifier(model: nn.Module, arch: str, num_classes: int) -> None:
    """Ganti lapisan classifier terakhir sesuai arsitektur."""
    if arch == "alexnet":
        # classifier AlexNet adalah nn.Sequential; lapisan terakhir indeks 6.
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif arch.startswith("resnet"):
        # ResNet: head klasifikasi bernama model.fc (satu nn.Linear).
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        # DenseNet: classifier adalah satu nn.Linear.
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)


def _build_densenet264_timm(num_classes: int, finetune: str) -> nn.Module:
    """DenseNet-264 pra-latih ImageNet via timm ("densenet264d").

    torchvision tidak menyediakan bobot ImageNet untuk DenseNet-264, jadi varian
    pra-latih dibangun lewat timm. timm membuat classifier baru berukuran
    `num_classes` secara langsung; untuk feature_extraction backbone dibekukan
    lalu head di-un-freeze kembali.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - bergantung environment
        raise ImportError(
            "densenet264 pra-latih memerlukan paket 'timm'. Pasang dengan "
            "`pip install timm`, atau pakai pretrained=False untuk DenseNet-264 "
            "dari nol via torchvision (lihat docs/ARSITEKTUR_BASELINE.md Bagian 4)."
        ) from exc

    model = timm.create_model(
        TIMM_PRETRAINED["densenet264"], pretrained=True, num_classes=num_classes
    )
    if finetune == "feature_extraction":
        _freeze_backbone(model)
        # Latih hanya head klasifikasi yang baru dibuat timm.
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Kembalikan (total_parameter, parameter_trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
