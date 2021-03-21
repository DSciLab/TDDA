PIP                 := pip
PYTHON              := python
REQUIREMENTS        := requirements.txt
CFG_PKG             := cfg
MLUTILS_PKG         := mlutils
VOXPY_PKG           := voxpy
JUNO_PKG            := juno
MAKE                := make
LAB_PORT            := 10012
DASHBOARD_PORT      := 20012
HOSTNAME            := 0.0.0.0
APP_DIR             := $(shell pwd)


.PHONY: install clean dep lab dashboard


install_cfg: $(CFG_PKG)
	$(MAKE) -C $^ install


install_juno: $(JUNO_PKG)
	$(MAKE) -C $^ install


install_mlutils: $(MLUTILS_PKG)
	$(MAKE) -C $^ install


install_voxpy: $(VOXPY_PKG)
	$(MAKE) -C $^ install


install: install_cfg install_mlutils install_voxpy install_juno


dep: $(REQUIREMENTS)
	$(PIP) install -r $^


lab:
	nohup jupyter lab --port $(LAB_PORT) --notebook-dir=$(APP_DIR) &


dashboard:
	nohup $(PYTHON) -m visdom.server -port $(DASHBOARD_PORT) --hostname $(HOSTNAME) &


clean:
	-rm -rf ./**/.tox ./**/.ipynb_checkpoints ./**/__pycache__ ./**/build
