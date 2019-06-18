init:
    pip install -r requirements.txt

run:
	python main.py

preprocess:
	python pre.py

.PHONY: init test