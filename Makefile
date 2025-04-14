setup:
	poetry lock && poetry install

test:
	poetry run pytest tests/ -v

install-pre-commit:
	@echo "Installing pre-commit..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "Detected macOS"; \
		which pre-commit > /dev/null || (echo "Installing pre-commit..." && brew install pre-commit && pre-commit install); \
	elif [ "$$(uname)" = "Linux" ]; then \
		echo "Detected Linux"; \
		which pre-commit > /dev/null || (echo "Installing pre-commit..." && sudo apt-get update && sudo apt-get install -y pre-commit && pre-commit install); \
	else \
		echo "Unsupported OS"; \
		exit 1; \
	fi
	@echo "pre-commit is installed."

format:
	@pre-commit run --all-files
	@echo "Just to be sure, we're running black again..."
	@poetry run black .

display-config:
	poetry run python evaluation.py -h

download-datasets:
	@echo "Downloading all available datasets..."
	@echo "Downloading earnings21 dataset..."
	@poetry run python common/download_dataset.py --dataset-name earnings21 --generate-only
	@echo "Downloading msdwild dataset..."
	@poetry run python common/download_dataset.py --dataset-name msdwild --generate-only
	@echo "Downloading icsi-meetings dataset..."
	@poetry run python common/download_dataset.py --dataset-name icsi-meetings --generate-only
	@echo "Downloading ali-meetings dataset..."
	@poetry run python common/download_dataset.py --dataset-name ali-meetings --generate-only
	@echo "Downloading aishell-4 dataset..."
	@poetry run python common/download_dataset.py --dataset-name aishell-4 --generate-only
	@echo "Downloading american-life dataset..."
	@poetry run python common/download_dataset.py --dataset-name american-life --generate-only
	@echo "Downloading ava-avd dataset..."
	@poetry run python common/download_dataset.py --dataset-name ava-avd --generate-only
	@echo "Downloading callhome dataset..."
	@poetry run python common/download_dataset.py --dataset-name callhome --generate-only
	@echo "All datasets have been downloaded successfully."
