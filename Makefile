PIP                 := pip
PYTHON              := python
REQUIREMENTS        := requirements.txt
CFG_PKG             := cfg
MLUTILS_PKG         := mlutils
VP_PKG              := vox_preprocessing
MAKE                := make
LAB_PORT            := 10012
DASHBOARD_PORT      := 20012
HOSTNAME            := 0.0.0.0
APP_DIR             := $(shell pwd)


.PHONY: install clean dep lab dashboard


install_cfg: $(CFG_PKG)
	$(MAKE) -C $^ install


install_mlutils: $(MLUTILS_PKG)
	$(MAKE) -C $^ install


install_vp: $(VP_PKG)
	$(MAKE) -C $^ install


install: install_cfg install_mlutils install_vp


dep: $(REQUIREMENTS)
	$(PIP) install -r $^


lab:
	nohup jupyter lab --port $(LAB_PORT) --notebook-dir=$(APP_DIR) &


dashboard:
	nohup $(PYTHON) -m visdom.server -port $(DASHBOARD_PORT) --hostname $(HOSTNAME) &


clean:
	-rm -rf .tox $(CFG_PKG)/.tox $(MLUTILS_PKG)/.tox .ipynb_checkpoints __pycache__
