# Modelling Passive Tracer Using PINNs

This repository presents a **Physics-Informed Neural Network (PINN)** approach to solving the **Advection-Diffusion Equation**, modeling the behavior of a passive tracer in a 2D periodic domain.

## 📌 Problem Formulation
The governing equation for the passive tracer is:


$\frac{\partial s}{\partial t} + \sin(y) \frac{\partial s}{\partial x} + \sin(x) \frac{\partial s}{\partial y} = 0.001 \nabla^2 s $

### **Initial Condition**
$s(x,y,0) = \sin(x) \sin(y)$

### **Boundary Conditions**
- Periodic in both \(x\) and \(y\)

## 📊 Results
The results include a **comparison between the spectral method and PINNs solution**:
  ![Comparison Results](results/comparison.gif)
<table>
  <tr>
    <td><img src="passive_tracer_spectral_pinns.gif" width="100%"/><br><center>Figure 1: dynamics of tracer.</center></td>
  </tr>
</table>

## 📂 Repository Structure
```
📦 Modelling_Passive_Tracer_Using_PINNs
│-- 📂 PINNs
│   ├── finetuning.py  # To fine-tune the trained model
│   ├── main_model.py  # Main PINNs model
│-- 📂 Spectral_method
│   ├── Spectral_method.py  # Solves using the spectral method
│   ├── plot.py  # Visualization of results
│-- README.md
```

## 🛠️ Requirements
To run this repository, install the following dependencies:
```bash
pip install deepxde dedalus h5py
```

## 🚀 Running the Models
### **Running the PINN Model**
```bash
python PINNs/main_model.py
```

### **Running the Spectral Method**
```bash
python Spectral_method/Spectral_method.py
```


