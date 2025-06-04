# TherMOS
TherMOS (Thermal Model Order Reduction and Simulation) focuses on the reduction of high-order
thermal models to more manageable lower-order models while preserving essential thermal dynamics. 
Thermal model order reduction is critical in simulations and real-time control applications where 
computational efficiency and accuracy are paramount. This process is typically necessary for systems 
with many thermal nodes, where full-scale modeling would be computationally prohibitive. It is clear, 
that this toolkit cannot be anywhere near the capabilities of commercial software but will hopefully 
provide a better understanding due to the freely available source code. The toolkit is obviously 
not complete; thus, suggestions are always welcome.

## Physical Model
The generalized architecture that should be modelled using the ROMs is displayed below. 
It aims to represent a generalized component in a power electronic system that is dissipating losses, 
while having boundary conditions on system level. The following aspects are considered:

1) Internal heat generation ($q_{ohmicHeatInt}^{*}$)
2) External heating due to nearby sources ($q_{ohmicHeat}^{*}$)
3) Heat flow to the ambient ($T_{amb}$)
4) Heat flow to the coolant ($T_{coolant}$) over the housing ($T_{housing}$)

Therefore, the full order model consists next, to the geometry, of two independent reference temperatures and losses. 

![phyMdl.png](docu%2Ffigures%2FphyMdl.png)

Figure 1: Physical Model describing the generalized architecture that should be modelled using the ROMs.

## Reduced Order Model
The aim of a Reduced Order Model (ROM) is to reduced a complex geometrical physical model
with high computational load using mathematical techniques. For example the temperature dependencies 
could then be modeled according to Fourier theory as a spatial temporal variable with infinite 
number of modes [1]:

$$ T(x, y, t) = \sum_{i=1}^{\infty} \theta_i(t) \, \phi_i(x, y) $$

Here:  
- $T(x, y, t)$ is the temperature distribution over space and time  
- $\phi_i(x, y)$ are the orthonormal spatial basis functions  
- $\theta_i(t)$ are the corresponding time-dependent coefficients (modes)

The model coefficients are determined by minimizing the reconstruction error between the full-order model and the reduced-order approximation:

$$ \min_{\phi_i(x, y)} \left\| T(x, y, t) - \hat{T}(x, y, t) \right\| $$

![romMdl.png](docu%2Ffigures%2FromMdl.png)

Figure 2: Example of ROM using a thermal network.


# Publication

"TherMOS: A Robust Low-Resolution 2D Spatial Thermal Model Order Reduction Approach"

The results presented in the above article can be directly reproduced using the setup files located in "TherMOS\setup\journal". To do so, the corresponding setup file needs to be copied to the root directory, i.e., "\TherMOS", and
executed from there. It must be noted that due to the size of the data, the data must first be generated for the results presented in Section IV-C/D, and is only pre-generated using 1 sec sampling resolution (instead of 100 ms)
for Section IV-B (it must be noted that these results are not the same as those presented in the paper due to the different sampling resolution). 

The data for exact reproduction of the results can be regenerated using the scripts stored in "TherMOS\gen\FEM\journal" and should then be placed under "TherMOS\data\journal".


# Dependencies
The requirements of the PyPowerSim toolkit are summarized below:
- Matlab R2023b
- identification_toolbox
- optimization_toolbox
- statistics_toolbox
- distrib_computing_toolbox
- image_toolbox
- neural_network_toolbox


# Datasets
The toolkit utilizes 1D and 2D datasets, as well as a large dataset for machine and deep learning
applications.

1) Tutorial RC (1D Foster Models) based on FE simulation
2) Tutorial SS (1D State-Space Models) based on FE simulation
3) Tutorial PO (2D POD) based on Matlab PDE generation (\gen)
4) Tutorial ML (Machine Learning) based on Electric Motor Temperature [1] (CC BY-SA 4.0): https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature
5) Tutorial DL (Deep Learning) based on Electric Motor Temperature [1] (CC BY-SA 4.0): https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature

If the user wants to utilize their own datasets, data can be provided in '.csv', '.xlsx', and '.mat' formats at the moment.
Data templates can be found under \data.


# Limitations
Since the toolkit is still under development there are several things that need to be 
improved, are not yet implemented or lack verification with numerical models or measurements.
In the following a list of know issues and limitations is provided:
- Numerical discrepancies for default implementation for calculating Gth and Cth matrices for 2D POD method
- Thermal structure function has convergence issues (should currently not be used)
- Deep learning does not support sequence-to-sequence learning


# Usage
Using TherMOS is straight forward and requires only three inputs, namely the model setup, the configuration file 
including experimental, data, model, as well as parameter information, and the training and testing data. Detailed 
information about these three inputs are provided below:

## Configuration 
The configuration file is an .xlsx file that is stored under \config and includes all parameter which are relevant for
the simulation. In detail, these parameters are grouped in four different categories, namely experimental (exp), data (Dat), 
model (Mdl), as well as parameters (Par). Explanation as well as valid inputs for each parameter can be found in the configuration file, 
description of the configuration file can be found below:

1) Name:          Descriptive name of the variable (arbitrary identifier)
2) Category:      Category of the variable (might be empty) 
3) Description:   Detailed description of the variable including valid options
4) Variable:      Variable name used inside the source code
5) Value:         Value of the respective variable
6) Unit:          Unit of the respective variable

From the above column only the 'Value' column should be adapted by the user, the other columns must not be changed as
they are used for reading the value.

## Start Script
To execute a new simulation three things are needed. First, a valid configuration file as discussed above. Second, valid
setup file and data. Third a start script as provided in start.m/start2D.m used for defining the configuration file as 
well as the model. The following aspects can be defined in the start script:

### General
1) Name:    Name of the simulation file (used for saving outputs)
2) User:    Author of the simulation
3) Debug:   Debug mode for checking internal results
4) Output:  Simulation output, i.e. output variable that is controlled (Current, Voltage, Power, etc.)
5) Type:    Simulation type, i.e. operating mode of the toolkit (Sweep, Steady-State, Transient, etc.)

### Input Files
1) Conf:    Configuration .xlsx file stored under \config used as input for general parameters and settings
2) Train:   Training data file (file based, sheet based, ID based, ratio based)
3) Test:    Testing data file (file based, sheet based, ID based, ratio based)

### Inputs and Output Mapping
1) Input:       List of input features, e.g. losses
2) Output:      List of output features, e.g. temperatures
3) Reference:   List of references, e.g. ambient or coolant temperature

### Plotting and Saving
1) Plot:    Plotting options for displaying results
2) PlotMdl: Plotting model structure
3) Save:    Option for saving results to \results


# Tutorials 
In the following chapter a set of reference results is provided using the 1D and 2D data examples.

## Data Descriptions

### 1D Model Order Reduction
The dataset comprises heat-up curves for a transformer with an E-core configuration, where thermal simulations are based 
on input losses for the primary winding, secondary winding, and core. Corresponding temperature responses are provided as 
output for each of these three components. The winding and core exhibit thermal coupling, with the degree of interaction 
varying depending on the specific operating point. To capture this behavior, measurements are included for three distinct 
operating points (OP1–OP3), reflecting different coupling and loss scenarios.

![trafo.png](docu%2Ffigures%2Ftrafo.png)
Figure 3: Example physical model using a transformer.

### 2D Model Order Reduction
The dataset includes thermal measurements from an insulated metal substrate (IMS) board, where heat is generated by 
two power semiconductor switches. A convective boundary condition is applied at the bottom surface, while all other 
edges are assumed adiabatic. Thermal coupling between the switches occurs through the shared copper layer, with the 
dominant heat flux directed toward the cooling boundary, reflecting realistic heat dissipation paths in power 
electronic modules.

![ims.png](docu%2Ffigures%2Fims.png)
Figure 4: Example physical model using a insulated metal substrate board.

## Results

### 1D-Results
State space models outperform traditional 1D RC (Foster) networks in terms of mean absolute error (MAE), mean squared 
error (MSE), and maximum error when modeling transformer thermal behavior. This is due to the inherent thermal coupling
between primary, secondary, and core losses, which mutually influence temperature distributions. While Foster networks 
fail to capture these interdependencies accurately, state space models inherently incorporate thermal coupling resistances 
and internal power distribution through their matrix structure, leading to more precise and robust temperature predictions.

| Model       | MAE (K) | MSE ($K^2$) | MAX (K) | 
|-------------|---------|-------------|---------|
| RC-Foster   | 17.3    | 336.3       | 22.7    |
| State-Space | 2.02    | 4.28        | 3.06    |

![results1D.png](docu%2Ffigures%2Fresults1D.png)
Figure 5: Error comparison for RC and SS models for the secondary winding.

### 2D-Results
The proposed model accurately captures both temperature and gradient distributions, with the largest temperature errors 
occurring at boundary interfaces and the highest gradient errors appearing at locations with sharp gradient transitions. 
These gradient deviations are directly linked to the thermal conductivity variations within the PCB materials. Temperature 
errors tend to increase in the direction of the heat sources, reflecting the complexity of heat propagation in those regions. 
Despite these challenges, the temporal behavior is well represented, with a maximum absolute error of approximately 1.0 K 
and a steady-state error around 0.2 K, demonstrating the model’s robustness and precision.

| Model  | MAE (K) | MSE ($K^2$) | MAX (K) | 
|--------|---------|-------------|---------|
| POD    | 0.86    | 0.82        | 2.15    |
| SS-POD | 0.64    | 0.41        | 1.75    |

![results2D.PNG](docu%2Ffigures%2Fresults2D.PNG)
Figure 6: Error comparison for POD models for the IMS board at the location of the heat source.


# Development
As failure and mistakes are inextricably linked to human nature, the toolkit is obviously not perfect, 
thus suggestions and constructive feedback are always welcome. If you want to contribute to the TherMOS 
toolkit or spotted any mistake, please contact me via: p.schirmer@herts.ac.uk

## Development Team
Dr. Pascal Schirmer, Dr.-Ing. Peter Schreivogel, Marcel Koenigseder 

## Future Work
The following aspects will be improved in future version:
- 3D input files for the Proper Orthogonal Decomposition (POD)
- Physics Informed Neural Networks (PINNs) for 1D and 2D problems
- Interface to Ansys


# License
The software framework is provided under the MIT License.


# Version History
1) v.0.0: (01.04.2023) Initial version of TherMOS
   

# References
[1] W. Kirchgässner, O. Wallscheid and J. Böcker, "Estimating Electric Motor Temperatures With Deep 
Residual Machine Learning," in IEEE Transactions on Power Electronics, vol. 36, no. 7, pp. 7480-7488, 
July 2021, doi: 10.1109/TPEL.2020.3045596
[2] Barabadi, Banafsheh, Satish Kumar, and Yogendra K. Joshi. "Transient heat conduction in on-chip interconnects using 
proper orthogonal decomposition method." Journal of Heat Transfer 139.7 (2017): 072101.
