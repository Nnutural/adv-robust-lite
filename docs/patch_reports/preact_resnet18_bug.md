PreActResNet-18 bug: inherited torchvision BasicBlock created bn1 for planes, but pre-activation applies bn1 before conv1 on in_planes.

Changed file: src/models/preact_resnet.py.

Validation:
- Forward shape check: pass, ok shape torch.Size([4, 10]), params 11172170.
- python -m pytest tests/test_models.py -q: pass by skip, 1 skipped because torchvision is not installed locally.
- python -m pytest tests/test_attacks_smoke.py -q: pass, 3 passed.
- python -m pytest tests/test_training_smoke.py -q: fail in local environment, ModuleNotFoundError: No module named 'torchvision'.
