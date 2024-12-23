from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
import torch
import dspy
from typing import List
from dataclasses import dataclass


class RewardEvaluator:
    """
    A simplified reward evaluator using a PRM model to score answers for a given question,
    integrated with dspy's BestOfNSignature for structured input and output.
    
    Example:
        evaluator = RewardEvaluator()
        signature = BestOfNSignature(
            problem="What is 2 + 2?",
            steps=["Step 1: Add 2 and 2."],
            answer="The answer is 4."
        )
        step_scores = evaluator.evaluate(signature)
        print(f"Step Scores: {step_scores}")
    """

    def __init__(
        self,
        model_name: str = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        device: str = "auto",
        torch_dtype=torch.bfloat16,
        **model_kwargs
    ):
        """
        Initializes the RewardEvaluator by loading the model and tokenizer.

        Args:
            model_name (str): The name or path of the pre-trained model.
            device (str): The device to map the model to. Default is "auto".
            torch_dtype (torch.dtype): The torch data type for the model.
            **model_kwargs: Additional keyword arguments for the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch_dtype,
            **model_kwargs,
        ).eval()

        # Set padding side and pad token
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # Encode "+" and "-" tokens
        plus_tag_id = self.tokenizer.encode("+", add_special_tokens=False)[-1]
        minus_tag_id = self.tokenizer.encode("-", add_special_tokens=False)[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

    def evaluate(self, problem, signature: dspy.Signature) -> List[float]:
        """
        Evaluates a single answer for a given problem statement and returns a list of step scores.

        Args:
            signature (BestOfNSignature): An instance containing the problem statement,
                                         steps, and the final answer.

        Returns:
            List[float]: A list of scores corresponding to each step in the answer.
        """
        steps = signature.steps

        step_scores = []
        conversation = []

        for k, step in enumerate(steps):
            if k == 0:
                # First step includes the problem statement
                text = f"{problem} {step}"
            else:
                text = step
            # Append user and assistant messages
            conversation.append({"content": text, "role": "user"})
            conversation.append({"content": "+", "role": "assistant"})

            # Prepare input_ids using the chat template
            input_ids = self.tokenizer.apply_chat_template(
                conversation, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                # Get logits for the candidate tokens at the specified position
                outputs = self.model(input_ids=input_ids)
                # Ensure that the model outputs logits
                if not hasattr(outputs, 'logits'):
                    raise AttributeError("Model output does not have 'logits' attribute.")
                logits = outputs.logits[:, -3, self.candidate_tokens]  # Adjust the position if necessary
                # Compute softmax probabilities
                step_score = logits.softmax(dim=-1)[:, 0]  # Probability of "+"
                # Append the score
                step_scores.append(
                    step_score[0].detach().cpu().item()
                )

        return step_scores

    def evaluate_multiple(self, signatures: List[dspy.Signature]) -> List[List[float]]:
        """
        Evaluates multiple answers and returns their respective lists of step scores.

        Args:
            signatures (List[BestOfNSignature]): A list of BestOfNSignature instances to evaluate.

        Returns:
            List[List[float]]: A list where each element is a list of step scores for the corresponding answer.
        """
        all_scores = []
        for signature in signatures:
            scores = self.evaluate(signature)
            all_scores.append(scores)
        return all_scores

