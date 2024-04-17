import transformers

class Trainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def train(self):
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4, # do an iteration of training, calculate gradients, but don;t update the weights, do another iteration, calculate gradients, add them to the previous iteration's gradient and then update the weights.
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            optim="paged_adamw_8bit",
            output_dir="fine_tuning",
            lr_scheduler_type="cosine", # Learning rate is critical to gradient descent : dynamic learning rate is best. Towards the end of training: small lr
            warmup_ratio=0.05,
            group_by_length=True, # process same length text together.
        )

        self.model.config.use_cache = False

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False), # data preprocessing
            args=training_args
        )

        trainer.train()