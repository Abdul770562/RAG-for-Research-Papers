"""
tests/test_ragas.py

A quick integration unit test to verify that the modern RAGAS metrics suite,
the LangchainLLMWrapper, and the EvaluatorLLM operate without throwing
type constraints or argument validation errors.
"""

import sys
from pathlib import Path

# Fix module resolution paths so utils and evaluation can be imported easily
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ragas import SingleTurnSample
from ragas.metrics.collections import ContextPrecision, ContextUtilization
from ragas.llms import LangchainLLMWrapper

from evaluation.evaluator_llm import EvaluatorLLM
from utils.logger import logger


def run_isolated_metric_test():
    logger.info("Starting isolated RAGAS smoke test...")

    try:
        # 1. Initialize our application core LLM infrastructure
        infra = EvaluatorLLM() 

        ragas_llm = infra.llm
        logger.info("Native InstructorLLM extracted successfully.")

        # 3. Build a pure mock single-turn data token matching your evaluation space
        mock_sample = SingleTurnSample(
            user_input="What problem does the Transformer architecture aim to solve?",
            response="The Transformer architecture aims to solve sequence transduction problems like machine translation without relying on recurrence.",
            reference="The Transformer architecture was proposed to solve sequence transduction tasks such as machine translation without relying on recurrent or convolutional networks.",
            retrieved_contexts=[
                "transduction problems such as language modeling and machine translation. Eschewing recurrence and instead relying entirely on an attention mechanism.",
                "Recurrent models typically factor computation along the symbol positions of the input and output sequences, preventing parallelization."
            ]
        )
        logger.info("Mock evaluation sample successfully constructed.")

        # 4. Instantiate target metrics collections
        precision_scorer = ContextPrecision(llm=ragas_llm)
        utilization_scorer = ContextUtilization(llm=ragas_llm)

        # 5. Execute single turn scoring runs synchronously
        logger.info("Executing row evaluation assertion (Context Precision)...")
        precision_score = precision_scorer.score(
            user_input=mock_sample.user_input,
            reference=mock_sample.reference,
            retrieved_contexts=mock_sample.retrieved_contexts
        )
        
        logger.info("Executing row evaluation assertion (Context Utilization)...")
        utilization_score = utilization_scorer.score(
            user_input=mock_sample.user_input,
            response=mock_sample.response,
            retrieved_contexts=mock_sample.retrieved_contexts
        )

        # 6. Verify and output tracking results
        print("\n" + "=" * 50)
        print("🎉 RAGAS INTEGRATION TEST PASSED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Context Precision Score  : {precision_score}")
        print(f"Context Utilization Score: {utilization_score}")
        print("=" * 50 + "\n")

    except Exception as e:
        logger.exception("Test execution collapsed with an active lifecycle fault!")
        print(f"\n❌ TEST FAILED: {str(e)}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_isolated_metric_test()