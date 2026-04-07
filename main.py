def main():
    notebooks = [
        "01_neurons_and_networks.py",
        "02_training_deep_networks.py",
        "03_convolutional_networks.py",
        "04_sequence_models.py",
        "05_attention_mechanism.py",
        "06_transformer_architecture.py",
        "07_language_models.py",
        "08_modern_llm_techniques.py",
    ]
    print("Deep Learning Marimo Notebooks")
    print("")
    print("Open a notebook with:")
    print("  marimo edit <notebook.py>")
    print("")
    print("Available notebooks:")
    for notebook in notebooks:
        print(f"  - {notebook}")


if __name__ == "__main__":
    main()
