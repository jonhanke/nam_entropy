.PHONY: show tree check env shell notebook webserver

## Check installed dependencies
show:
	poetry show

## Show dependency tree
tree:
	poetry show --tree

## Validate pyproject.toml
check:
	poetry check

## Show Poetry environment info
env:
	poetry env info

## Activate Poetry shell
shell:
	poetry shell

## Run the Notebook for this Poetry virtual environment
notebook:
	poetry run jupyter notebook

