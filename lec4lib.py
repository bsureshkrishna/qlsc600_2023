{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNPn+J8Z2G8bxfT8EDLkm9v",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/bsureshkrishna/qlsc600_2023/blob/main/lec4lib.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bzZrWWJlF7l4"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from scipy.signal import butter, filtfilt\n",
        "from scipy.stats import gamma\n",
        "import math\n",
        "\n",
        "def plot_signal_spectrum(signal, fs, plotornot = 1):\n",
        "    # Compute the FFT\n",
        "    fft_result = np.fft.fft(signal-np.mean(signal))\n",
        "\n",
        "    # Compute the frequency bins\n",
        "    freqs = np.fft.fftfreq(len(signal), d=1/fs)\n",
        "\n",
        "    # Compute the magnitude (absolute value of the FFT)\n",
        "    magnitude = np.abs(fft_result)\n",
        "\n",
        "    if(plotornot==1):\n",
        "        # Plotting\n",
        "        plt.plot(freqs, magnitude)\n",
        "        plt.xlabel(\"Frequency (Hz)\")\n",
        "        plt.ylabel(\"Magnitude\")\n",
        "        plt.title('Spectrum')\n",
        "\n",
        "    return freqs, magnitude"
      ]
    }
  ]
}