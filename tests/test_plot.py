import pytest
import matplotlib.pyplot as plt

@pytest.mark.mpl_image_compare(baseline_dir='baseline', filename='other_name.png')
def test_succeeds():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([1,2,3])
    return fig
