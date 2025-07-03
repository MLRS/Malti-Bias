## How to Generate Prompts and Apply AutoDebias
Code based on [this paper](https://aclanthology.org/2022.acl-long.72/).

To generate prompts, use the following command:

```bash
!python generate_prompts.py --debias_type gender --model_type bert --model_name_or_path bert-base-cased
```

To apply the AutoDebias technique, use the following command:

```bash
!python auto-debias.py --debias_type gender --model_type bert --model_name_or_path bert-base-uncased --prompts_file prompts_bert-base-uncased_gender
```

### Command Arguments:
- `--debias_type`: The type of debiasing to apply (e.g., `gender`).
- `--model_type`: The type of model being used (e.g., `bert`).
- `--model_name_or_path`: The model's name or path (e.g., `bert-base-cased`, `bert-base-uncased`).
- `--prompts_file`: The file containing prompts for the model (e.g., `prompts_bert-base-uncased_gender`).
