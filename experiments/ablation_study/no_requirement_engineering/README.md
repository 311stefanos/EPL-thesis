# Personal Receipt Agent

This project contains the receipt-processing graph and its prompts.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure `personal_receipt_agent/.env` with the credentials and settings expected by the external model/subgraph integration. Do not commit secrets.
4. Import or run the application entry point used by your host integration. The receipt ledger is created automatically when a receipt is successfully saved; no seed data is included.

The supplied Python modules are intentionally unchanged. `personal_receipt_agent/__init__.py` makes the directory importable as a package.
