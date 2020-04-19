from collections import namedtuple

HyperParameter = namedtuple("HyperParameter",
                            field_names=["learning_rate", "adam_epsilon", "hidden_size", "rollout_length",
                                         "discount_rate", "lambda_", "mini_batch_number", "optimization_epochs",
                                         "ppo_clip",
                                         "entropy_coefficent", "gradient_clip"])

HyperParameter.__new__.__defaults__ = (3e-4, 1e-5, 512, 500, 0.99, 0.95, 32, 10, 0.2, 0.01, 5)
