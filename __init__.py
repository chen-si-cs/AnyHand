"""
AnyHand — unified hand pose estimation predictor.

Quick start
-----------
>>> from anyhand import AnyHandPredictor, HandPrediction
>>> predictor = AnyHandPredictor(backend='wilor')        # or 'hamer' / 'both'
>>> results = predictor.predict('path/to/image.jpg')
>>> for hand in results:
...     print(hand.is_right, hand.mano_pose.shape, hand.vertices.shape)
"""

from anyhand.predictor import AnyHandPredictor, HandPrediction

__all__ = ['AnyHandPredictor', 'HandPrediction']
