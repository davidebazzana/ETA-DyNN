# ETA-DyNN

This is repository contains the code for the paper "**ETA-DyNN: An Energy-Efficient Task Allocation Scheme for Varroa Detection on Edge AI Devices via Dynamic Neural Networks**".

You can find four folders:
- `activity_emulator`: an interactive interface to test the ETA-DyNN approach;
- `environment_aware_task_allocation`: the proposed ETA-DyNN decision-making module;
- `ee_cnn`: the Early-Exit NN;
- `key_frame_extraction`: the algorithm that extracts key frames from the input video.

In order to run any of the code, you first need to
1. install the packages locally through:
```pip install -e .```
2. install the required python dependencies:
```pip install -r requirements.txt```

Then you can move to the folder `activity_emulator` and run the experiment interface with `python main.py`.

In the following, we report the complete description of the formulas used to compute the power output of the PV module.

## Environmental Dataset

The system functionality depends heavily on environmental conditions. We assume the system is deployed in rural areas and off-grid energy settings. Therefore, the system must harvest the energy it needs on-site, and the most common way to do so is with Photovoltaic (PV) panels. This type of energy generation depends on several environmental factors (Huld et al., 2011), such as:

- solar irradiance;
- module temperature;
- module surface reflectivity and angle-of-incidence effect;
- spectral effects.

To evaluate the system in a realistic environment without deploying it in the field, we implemented an emulation environment to replicate real PV panel power outputs, inspired by the PV-GIS tool (Suri et al., 2005; European Commission, JRC PVGIS tools). The emulation environment takes as input sequences of environmental data, including Global Tilted Irradiance (GTI), Global Horizontal Irradiance (GHI), solar zenith and azimuth, air temperature, and wind (at ground level), and computes the power output of the PV system.

We set all coefficients of the formulas defining the power output of a PV module ($a_r, U_0, U_1, k_{1..6}$) (Huld et al., 2011) as in PVGIS data sources and calculation methods (European Commission, JRC).

---

### Angular losses

We compute the angular losses of the PV panel as in Martin and Ruiz (2001):

$$
AL = 1 - \frac{1 - \exp\left(-\cos\left(\frac{aoi}{a_r}\right)\right)}{1 - \exp\left(-\frac{1}{a_r}\right)}
$$

where `aoi` is the angle of incidence of the radiation on the PV module surface and $a_r$ is the *angular losses coefficient*.

We define the irradiance after angular losses as:

$$
G_{AL} = (1 - AL)\cdot G
$$

---

### Module temperature

We compute the module temperature as in Faiman (2008):

$$
T_{mod} = T_{amb} + \frac{G}{U_0 + U_1 \cdot W}
$$

where $T_{amb}$ is the air temperature, $W$ is the wind speed, and $U_0, U_1$ are empirical coefficients.

---

### Power output model

Finally, we compute the power output of the PV module as (Huld et al., 2011):

$$
\begin{aligned}
P &= G' \cdot \bigg(P_{mod_{STC}} + k_1 \ln(G') + k_2 \ln^2(G') + k_3 T' \\
&\quad + k_4 T' \ln(G') + k_5 T' \ln^2(G') + k_6 T'^2 \bigg)
\end{aligned}
$$

where:

- $G' = G_{AL} / G_{STC}$
- $T' = T_{mod} - T_{mod_{STC}}$

---

Spectral effects are excluded from the simulation due to the inherent difficulty of modeling the radiation spectrum. Moreover, they would only result in a slight change in the overall estimated energy output (Huld et al., 2011).

---

### Data source and configuration

We use environmental data provided by :contentReference[oaicite:0]{index=0}. The dataset corresponds to the location:

- 40.71720°N, 8.55406°E (Department of Agriculture, University of Sassari, where the Varroa Destructor Video Dataset was captured).

For the GTI parameter, we configure the PV module as follows:

- fixed tracker type
- 35° fixed tilt angle
- 180° fixed azimuth (optimized values for the above location)

---

## References

- Huld, T., Friesen, G., Skoczek, A., Kenny, R. P., Sample, T., Field, M., & Dunlop, E. D. (2011). *A power-rating model for crystalline silicon PV modules*. Solar Energy Materials and Solar Cells, 95(12), 3359–3369.

- Suri, M., Huld, T. A., & Dunlop, E. D. (2005). *PV-GIS: a web-based solar radiation database for the calculation of PV potential in Europe*. International Journal of Sustainable Energy, 24(2), 55–67.

- European Commission, Joint Research Centre (JRC). *Photovoltaic Geographical Information System (PVGIS) — PV Tools*. https://re.jrc.ec.europa.eu/pvg_tools/en/

- European Commission, Joint Research Centre (JRC). *PVGIS Data Sources & Calculation Methods*. https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/pvgis-data-sources-calculation-methods_en

- Martin, N., & Ruiz, J. M. (2001). *Calculation of the PV modules angular losses under field conditions by means of an analytical model*. Solar Energy Materials and Solar Cells, 70(1), 25–38.

- Faiman, D. (2008). *Assessing the outdoor operating temperature of photovoltaic modules*. Progress in Photovoltaics: Research and Applications, 16(4), 307–315.
