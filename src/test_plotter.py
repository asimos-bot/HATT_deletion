import pytest
from plotter import Plotter

path_to_log = '../log.log'

@pytest.fixture
def plotter_obj():

    return Plotter(path_to_log)

def test_plotter_creation():

    try:
        plotter = Plotter(path_to_log)
    except:
        assert False

def test__field_has_keyword(plotter_obj):

    field = "acc_mean_[HATT 0%]"
    keywords = ["id", "mean"]

    assert plotter_obj._field_has_keyword(field, keywords) == True
    assert plotter_obj._field_has_keyword(field, ["giraffe"]) == False
    assert plotter_obj._field_has_keyword("", keywords) == False

def test__filter_headers(plotter_obj):

    assert plotter_obj._filter_headers(["id", "mean"]) == {"mean_acc_[HATT 75%]", "id", "mean_acc_[HATT 0%]", "mean_acc_[HATT 10%]", "mean_acc_[HATT 25%]", "mean_acc_[HATT 50%]"}
