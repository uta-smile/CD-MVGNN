import json

from nose.tools import eq_

from third_party.mordred import Calculator
from third_party.mordred import descriptors


def test_json():
    calc = Calculator(descriptors)
    j = json.dumps(calc.to_json())
    calc2 = Calculator.from_json(json.loads(j))
    eq_(calc.descriptors, calc2.descriptors)
