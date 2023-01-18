import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

display_results = True

xpoints = [5, -5, 1, -1, -3, 3, 4, -4] # Rating differential (Home - Away)
ypoints = [1, 0, 0, 1, 0, 1, 1, 0] # Home Win/Loss Boolean (Win = 1, Tie = 0.5, Loss = 0)

print(xpoints)

for index in range(len(xpoints)):
    xpoints[index] *= -1

print(xpoints)

parameters, covariates = curve_fit(lambda t, param: 1/(1+np.exp((t)/param)), xpoints, ypoints)
param = parameters[0]

if display_results == True:
    x_fitted = np.linspace(np.min(xpoints), np.max(xpoints), 100)
    y_fitted = 1/(1+np.exp((x_fitted)/param))

    r, p = pearsonr(xpoints, ypoints)

    if p > 0.1: significance = 'No' 
    elif p > 0.05: significance = 'Weak'
    elif p > 0.05: significance = 'Weak'
    elif p > 0.01: significance = 'Moderate'
    elif p > 0.005: significance = 'Good'
    elif p > 0.001: significance = 'Strong'
    else: significance = 'Very strong'

    print(f'\nPearson Correlation of Independent and Dependent Variables: {r:.3f}')
    print(f'Significance of Correlation (p-value): {p:.5f}\t({significance} evidence against the null hypothesis)')
    print(f'R² of Regressed Sigmoid Function: {r2_score(ypoints, 1/(1+np.exp((xpoints)/param))):.3f} | 1/(1+exp((x)/{param:.3f}))')

    plt.plot(xpoints, ypoints, 'o', color='grey')
    plt.plot(x_fitted, y_fitted, color='black', alpha=1, label=f'Sigmoid (R² = {r2_score(ypoints, 1/(1+np.exp((xpoints)/param))):.3f})')
    plt.legend()
    plt.title('Logistic Regression of Team Rating vs Game Results')
    plt.xlabel('Rating Difference')
    plt.ylabel('Game Result (Binary)')
    plt.show()