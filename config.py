class Config:
    # Model Configuration
    MODEL_NAME = 'microsoft/deberta-v3-base'
    NUM_LABELS = 2  # Binary classification (spam/ham)
    MAX_LENGTH = 512  # Max sequence length for tokenizer
    
    # Training Configuration
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Paths
    DATA_PATH = 'data/enron_emails_processed.csv'
    # Point to the local directory that contains the saved DeBERTa model/tokenizer
    MODEL_SAVE_PATH = 'models/deberta_email_classifier'
    
    # Training/Validation Split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
