import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from scipy.integrate import solve_ivp
import os
import glob
import math
import time
from sklearn.metrics import mean_squared_error

class CompostingModel:
    def __init__(self, df):
        self.df_combined = df
        self.initial_conditions_list = []
        self.kinetic_parameters = self.set_kinetic_parameters()
        self.molecular_weights = [180.16, 336.403, 381.645, 246.268, 354.464, 180.156]
        # Initialize tracking for gas emissions and compost production
        self.cumulative_emissions = {
            'CO2': 0.0,  # kg
            'CH4': 0.0,  # kg
            'N2O': 0.0,  # kg
            'NH3': 0.0   # kg
        }
        self.compost_production = {
            'mass': TM,
            'quality_parameters': {
                'moisture_content': None,
                'organic_matter': None,
                'cn_ratio': None
            }
        }
    def set_kinetic_parameters(self):
        return {
            'Kh_Xc_M': 0.040, 'Kh_Xp_M': 0.020, 'Kh_L_XMB_OR_XTB_3': 0.010,
            'Kh_Xc_T': 0.020, 'Kh_Xp_T': 0.040, 'Kh_H': 0.02, 'Kh_CE': 0.02,
            'Kh_LG': 0.02, 'Kh_S': 1.5, 'µ_MB': 0.2, 'µ_TB': 0.18,
            'µ_MA_F': 0.1, 'µ_TA_F': 0.12, 'K_S': 62, 'K_O2': 0.07,'K_NH4': 0.05, 'b_MB': 0.01,
            'b_TB': 0.01, 'b_MA_F': 0.01, 'b_TA_F': 0.007, 'K_dec': 0.0025,
            'Kl_O2': 1.00E-4, 'Kl_CO2': 1.00E-4, 'Kl_NH3': 1.00E-4, 'Kl_H2O_v': 1.00E-4,
            'Y_O2_X_C': 0.562, 'Y_O2_X_P': 1.065527402, 'Y_O2_X_L': 2.304874433,
            'Y_O2_X_H': 0.795378756, 'Y_O2_X_LG': 1.72835087, 'Y_CO2_X_C': 0.772546553,
            'Y_CO2_X_P': 1.313922605, 'Y_CO2_X_L': 2.189731603, 'Y_CO2_X_H': 1.153858305,
            'Y_CO2_X_LG': 1.849966693, 'Y_NH3_X_C': -0.054, 'Y_NH3_X_P': 0.148883698,
            'Y_NH3_X_L': -0.053659085, 'Y_NH3_X_H': -0.024508647, 'Y_NH3_X_LG': -0.024508647,
            'Y_H2O_X_C': 0.460, 'Y_H2O_X_P': 0.054, 'Y_H2O_X_L': 0.949, 'Y_H2O_X_H': 1.470,
            'Y_H2O_X_LG': 0.581
        }

    def calculate_initial_conditions(self):
        for index, row in self.df_combined.iterrows():
            W_0 = row['Mc']
            X_P_0 = row['P']
            X_L_0 = row['L']
            X_C_0 = row['C']
            X_CE_0 = row['CE']
            X_H_0 = row['H']
            X_LG_0 = row['LG']
            X_A_0 = row['A']
            X_In_0 = row['In']
            X_F_0 = X_CE_0 + X_H_0 + X_LG_0
            self.TM_0 = row.sum()
            Mc_0 = W_0 / self.TM_0
            S_C_0 = 0.5
            S_P_0 = 0.5
            S_L_0 = 0.5
            S_H_0 = 0.5
            S_CE_0 = 0.5
            S_LG_0 = 0.5
            X_MB_0 = 1.8e-3 * (self.TM_0 - W_0)
            X_TB_0 = 1.4e-4 * (self.TM_0 - W_0)
            X_MA_F_0 = 1.8e-3 * (self.TM_0 - W_0)
            X_TA_F_0 = 1.4e-4 * (self.TM_0 - W_0)
            X_DB_0 = 0
            T_0 = 20
            O2_0 = 0.5
            CO2_0 = 0.01
            CH4_0 = 0.01
            NH3_0 = 0.01
            H2O_0 = 0.01
            CO2_atm_0 = 0
            NH3_atm_0 = 0
            H20_atm_0 = 0

            C_C_0 = (X_C_0 / self.molecular_weights[0]) * 6
            C_P_0 = (X_P_0 / self.molecular_weights[1]) * 16
            C_L_0 = (X_L_0 / self.molecular_weights[2]) * 25
            C_H_0 = (X_H_0 / self.molecular_weights[3]) * 10
            C_CE_0 = (X_CE_0 / self.molecular_weights[5]) * 6
            C_LG_0 = (X_LG_0 / self.molecular_weights[4]) * 20

            initial_conditions = [
                X_MB_0, X_TB_0, X_MA_F_0, X_TA_F_0, X_DB_0, W_0, X_P_0, X_L_0, X_C_0, X_H_0,
                X_CE_0, X_LG_0, X_A_0, X_In_0, S_C_0, S_P_0, S_L_0, S_H_0, S_CE_0, S_LG_0,
                self.TM_0, T_0, O2_0, CO2_0, CH4_0, NH3_0, H2O_0, CO2_atm_0, NH3_atm_0, H20_atm_0,
                C_C_0, C_P_0, C_L_0, C_H_0, C_CE_0, C_LG_0
            ]
            self.initial_conditions_list.append(initial_conditions)
        self.N2O = ((-0.260*(Mc_0*100) + 29.38)/1000)*(1-Mc_0)*(self.TM_0/1000) # from https://doi.org/10.11449/sasj.42.1_18 (the equation in the tabel was -0.260*Mc - 29.38, wrongly written (the - instead of +))
    def calculate_CN_ratio(self, X_C_0, X_P_0, X_L_0, X_H_0, X_CE_0, X_LG_0):
        C_carbohydrate = (X_C_0 / self.molecular_weights[0]) * 6
        C_proteins = (X_P_0 / self.molecular_weights[1]) * 16
        C_lipids = (X_L_0 / self.molecular_weights[2]) * 25
        C_hemycellulose = (X_H_0 / self.molecular_weights[3]) * 10
        C_cellulose = (X_CE_0 / self.molecular_weights[5]) * 6
        C_lignin = (X_LG_0 / self.molecular_weights[4]) * 20
        N_proteins = (X_P_0 / self.molecular_weights[1]) * 4
        total_carbon = C_carbohydrate + C_proteins + C_lipids + C_hemycellulose + C_cellulose + C_lignin
        CN_feed = total_carbon / N_proteins
        return CN_feed

    def dX_dt(self, t, X):
        X_MB, X_TB, X_MA_F, X_TA_F, X_DB, W, X_P, X_L, X_C, X_H, X_CE, X_LG, X_A, X_In, S_C, S_P, S_L, S_H, S_CE, S_LG, TM, T, O2, CO2, CH4, NH3, H2O, CO2_atm, NH3_atm, H20_atm, C_C, C_P, C_L, C_H, C_CE, C_LG = X

        ρ_C = 1591.3
        ρ_P = 1316.9
        ρ_L = 915.1
        ρ_F = 1302.3
        ρ_A = 2416.78
        ρ_In = 1000
        ρ_H2O = 998
        ρ_p = 1365.7
        X_F = X_CE + X_H + X_LG
        Mc = W / TM
        Sc = (TM - W) / TM
        Lc = (X_L + S_C) / TM
        density = 25000 / 30
        V = TM / density
        b = 1
        B = 3
        h = 1.5
        L = V / ((h * (b + B)) / 2)
        a = math.sqrt(((B - b) / 2) ** 2 + h ** 2)
        A = h * (b + B) + L * (2 * a + b + B)
        A_cross = ((b + B) / 2) * h
        FAS = 1 - (Sc * density / ρ_p) - (Mc * density / ρ_H2O) - (Lc * density / ρ_L)
        R = 8.314
        g = 9.81
        P = 101325
        T_amb = 298.15
        P_sat_amb = 2339
        R_spec = 287.058
        R_v = 461.495
        T_K = T + 273.15
        T_0 = 273.15
        P_sat = 10 ** (10.196213 - (1730.63 / (T + 233.426)))
        ρ_air_amb = ((P - P_sat_amb) / (R_spec * T_amb)) + (P_sat_amb / (R_v * T_amb))
        ρ_air_compost = ((P - P_sat) / (R_spec * T_K)) + (P_sat / (R_v * T_K))
        ΔP = (ρ_air_amb - ρ_air_compost) * g * h
        dp = 0.0155
        Er = 180
        K = ((dp ** 2) * (FAS ** 3)) / (Er * ((1 - FAS) ** 2))
        µ_air = (1.716 * 10 ** -5) * ((T_K / T_0) ** 1.5) * ((T_0 + 111) / (T_K + 111))
        V_air = (K * A_cross * ΔP) / (h * µ_air)
        m_air = V_air * ρ_air_compost
        V_O2 = V_air * 0.21
        m_O2 = V_O2 * ρ_air_compost

        nO2 = O2 / 32
        nCO2 = CO2 / 44
        nNH3 = NH3 / 17
        nH2O = H2O / 18
        X_O2 = nO2 / (nO2 + nCO2 + nNH3)
        X_CO2 = nCO2 / (nO2 + nCO2 + nNH3)
        X_NH3 = nNH3 / (nO2 + nCO2 + nNH3)

        He_O2 = 42014
        He_CO2 = 1606.42
        He_NH3 = 0.925732

        p34 = self.kinetic_parameters['Kl_O2'] * (He_O2 * X_O2 - (0.20946 * 101325))
        p35 = self.kinetic_parameters['Kl_CO2'] * (He_CO2 * X_CO2 - (0.000412 * 101325))
        p36 = self.kinetic_parameters['Kl_NH3'] * (He_NH3 * X_NH3 - (0.00000179 * 101325))
        p37 = self.kinetic_parameters['Kl_H2O_v'] * P_sat

        U = 4180
        A = h * (b + B) + L * (2 * a + b + B)
        q_conv = (U * A * (T - 20))
        X_C = max(0, X_C)
        X_P = max(0, X_P)
        X_L = max(0, X_L)
        X_H = max(0, X_H)
        X_CE = max(0, X_CE)
        X_LG = max(0, X_LG)
        S_C = max(0, S_C)
        S_P = max(0, S_P)
        S_L = max(0, S_L)
        S_H = max(0, S_H)
        S_CE = max(0, S_CE)
        S_LG = max(0, S_LG)
        T_sporulation = 55
        p1 = self.kinetic_parameters['Kh_Xc_M'] * (X_MB / (self.kinetic_parameters['Kh_S'] * X_MB + X_C)) * X_C if T < T_sporulation and X_C > 0 else 0
        p2 = self.kinetic_parameters['Kh_Xp_M'] * (X_MB / (self.kinetic_parameters['Kh_S'] * X_MB + X_P)) * X_P if T < T_sporulation and X_P > 0 else 0 
        p3 = self.kinetic_parameters['Kh_L_XMB_OR_XTB_3'] * (X_MB / (self.kinetic_parameters['Kh_S'] * X_MB + X_L)) * X_L if T < T_sporulation and X_L > 0 else 0
        p4 = self.kinetic_parameters['Kh_Xc_T'] * (X_TB / (self.kinetic_parameters['Kh_S'] * X_TB + X_C)) * X_C if  X_C > 0 else 0
        p5 = self.kinetic_parameters['Kh_Xp_T'] * (X_TB / (self.kinetic_parameters['Kh_S'] * X_TB + X_P)) * X_P if X_P > 0 else 0
        p6 = self.kinetic_parameters['Kh_L_XMB_OR_XTB_3'] * (X_TB / (self.kinetic_parameters['Kh_S'] * X_TB + X_L)) * X_L if X_L > 0 else 0 
        p7 = self.kinetic_parameters['Kh_H'] * (X_MA_F / (self.kinetic_parameters['Kh_S'] * X_MA_F + X_H)) * X_H if T < T_sporulation and X_H > 0 else 0
        p8 = self.kinetic_parameters['Kh_H'] * (X_TA_F / (self.kinetic_parameters['Kh_S'] * X_TA_F + X_H)) * X_H if X_H > 0 else 0
        p9 = self.kinetic_parameters['Kh_CE'] * (X_MA_F / (self.kinetic_parameters['Kh_S'] * X_MA_F + X_CE)) * X_CE if T < T_sporulation and X_CE > 0 else 0
        p10 = self.kinetic_parameters['Kh_LG'] * (X_MA_F / (self.kinetic_parameters['Kh_S'] * X_MA_F + X_LG)) * X_LG if T < T_sporulation and X_LG > 0 else 0
        p11 = self.kinetic_parameters['Kh_CE'] * (X_TA_F / (self.kinetic_parameters['Kh_S'] * X_TA_F + X_CE)) * X_CE if  X_CE > 0 else 0
        p12 = self.kinetic_parameters['Kh_LG'] * (X_TA_F / (self.kinetic_parameters['Kh_S'] * X_TA_F + X_LG)) * X_LG if X_LG > 0 else 0

        T_min = 5
        T_max = 55
        T_opt = 37
        if T <= T_min:
            f_T1 = 0
        elif T <= T_max:
            f_T1 = ((T - T_max) * ((T - T_min) ** 2)) / ((T_opt - T_min) * ((T_opt - T_min) * (T - T_opt) - (T_opt - T_max) * (T_opt + T_min - (2 * T))))
        else:
            f_T1 = 0

        T_min_1 = 30
        T_max_1 = 70
        T_opt_1 = 55
        if T <= T_min_1:
            f_T2 = 0
        elif T <= T_max_1:
            f_T2 = ((T - T_max_1) * ((T - T_min_1) ** 2)) / ((T_opt_1 - T_min_1) * ((T_opt_1 - T_min_1) * (T - T_opt_1) - (T_opt_1 - T_max_1) * (T_opt_1 + T_min_1 - (2 * T))))
        else:
            f_T2 = 0

        f_SC = S_C / (self.kinetic_parameters['K_S'] + S_C)
        f_SL = S_L / (self.kinetic_parameters['K_S'] + S_L)
        f_SP = S_P / (self.kinetic_parameters['K_S'] + S_P)
        f_SH = S_H / (self.kinetic_parameters['K_S'] + S_H)
        f_SLG = S_LG / (self.kinetic_parameters['K_S'] + S_LG)

        f_O2 = (O2 / V) / (self.kinetic_parameters['K_O2'] + (O2 / V))
        #ammonia 
        NH3 = max(0, NH3)
        f_NH4 = NH3/(self.kinetic_parameters['K_NH4']+NH3)
        X_Mc = W / TM
        f_IW = 1 / (np.exp(-17.684 * X_Mc + 7.0622) + 1)

        f_FAS = 1 / (math.exp(-23.675 * FAS + 3.4945) + 1)

        p13 = self.kinetic_parameters['µ_MB'] * f_T1 * f_O2 * f_IW * f_SC * f_FAS *f_NH4* X_MB if T < T_sporulation else 0.01
        p14 = self.kinetic_parameters['µ_MB'] * f_T1 * f_O2 * f_IW * f_SP * f_FAS * X_MB if T < T_sporulation else 0.01
        p15 = self.kinetic_parameters['µ_MB'] * f_T1 * f_O2 * f_IW * f_SL * f_FAS *f_NH4* X_MB if T < T_sporulation else 0.01
        p16 = self.kinetic_parameters['µ_TB'] * f_T2 * f_O2 * f_IW * f_SC * f_FAS *f_NH4* X_TB
        p17 = self.kinetic_parameters['µ_TB'] * f_T2 * f_O2 * f_IW * f_SP * f_FAS * X_TB
        p18 = self.kinetic_parameters['µ_TB'] * f_T2 * f_O2 * f_IW * f_SL * f_FAS *f_NH4* X_TB
        p19 = self.kinetic_parameters['µ_MA_F'] * f_T1 * f_O2 * f_IW * f_SC * f_FAS *f_NH4* X_MA_F if T < T_sporulation else 0.01
        p20 = self.kinetic_parameters['µ_MA_F'] * f_T1 * f_O2 * f_IW * f_SP * f_FAS * X_MA_F if T < T_sporulation else 0.01
        p21 = self.kinetic_parameters['µ_MA_F'] * f_T1 * f_O2 * f_IW * f_SL * f_FAS *f_NH4* X_MA_F if T < T_sporulation else 0.01
        p22 = self.kinetic_parameters['µ_MA_F'] * f_T1 * f_O2 * f_IW * f_SH * f_FAS *f_NH4* X_MA_F if T < T_sporulation else 0.01
        p23 = self.kinetic_parameters['µ_MA_F'] * f_T1 * f_O2 * f_IW * f_SLG *f_NH4* X_MA_F
        p24 = self.kinetic_parameters['µ_TA_F'] * f_T2 * f_O2 * f_IW * f_SC *f_NH4* f_FAS * X_TA_F
        p25 = self.kinetic_parameters['µ_TA_F'] * f_T2 * f_O2 * f_IW * f_SP * f_FAS * X_TA_F
        p26 = self.kinetic_parameters['µ_TA_F'] * f_T2 * f_O2 * f_IW * f_SL *f_NH4* f_FAS * X_TA_F
        p27 = self.kinetic_parameters['µ_TA_F'] * f_T2 * f_O2 * f_IW * f_SH *f_NH4* f_FAS * X_TA_F
        p28 = self.kinetic_parameters['µ_TA_F'] * f_T2 * f_O2 * f_IW * f_SLG *f_NH4* X_TA_F

        p29 = (self.kinetic_parameters['b_MB'] * X_MB)
        p30 = (self.kinetic_parameters['b_TB'] * X_TB)
        p31 = (self.kinetic_parameters['b_MA_F'] * X_MA_F)
        p32 = (self.kinetic_parameters['b_TA_F'] * X_TA_F)
        X_DB = p29 + p30 + p31 + p32

        p33 = (self.kinetic_parameters['K_dec'] * X_DB)

        dC_C_dt = - (p1 + p4) * (6 / self.molecular_weights[0])
        dC_P_dt = - (p2 + p5) * (16 / self.molecular_weights[1])
        dC_L_dt = - (p3 + p6) * (25 / self.molecular_weights[2])
        dC_H_dt = - (p7 + p8) * (10 / self.molecular_weights[3])
        dC_CE_dt = - (p9 + p11) * (6 / self.molecular_weights[5])
        dC_LG_dt = - (p10 + p12) * (20 / self.molecular_weights[4])
        C_consumed = -(dC_C_dt + dC_P_dt + dC_L_dt + dC_H_dt + dC_CE_dt + dC_LG_dt)

        Mc_1 = Mc * 100
        if Mc_1 <= 70:
            #log10_CH4 = -9.43 + 12.14 * Mc
            #CH4_emission_factor = 10 ** log10_CH4

            CH4_emission_factor = math.exp(-16.6771 + 0.274338 * Mc_1)
        else:
            CH4_emission_factor = 20  # for MC >= 70%

        dW_dt = -p37
        dX_MB_dt = (p13 + p14 + p15) - p29
        dX_TB_dt = (p16 + p17 + p18) - p30
        dX_MA_F_dt = (p19 + p20 + p21 + p22) - p31
        dX_TA_F_dt = (p24 + p25 + p26 + p27) - p32
        dX_DB_dt = p33
        dX_C_dt = - (p1 + p4)
        dX_P_dt = - (p2 + p5)
        dX_L_dt = - (p3 + p6)
        dX_H_dt = - (p7 + p8)
        dX_CE_dt = - (p9 + p11)
        dX_LG_dt = -(p23 + p28)
        dX_A_dt = 0
        dX_In_dt = 0
        dS_C_dt = (p1 + p4 + p9 + p11 - (p13 + p16 + p19 + p24))
        dS_CE_dt = 0
        dS_P_dt = (p2 + p5 + p33 - (p14 + p17 + p20 + p25))
        dS_L_dt = (p3 + p6 - (p15 + p18 + p21 + p26))
        dS_H_dt = (p7 + p8 - (p22 + p27))
        dS_LG_dt = (p10 + p12 - (p23 + p28))
        #dTM_dt = -(p35 + p36 + p37)
        dO2_dt = -((p13 + p16 + p19 + p24) * self.kinetic_parameters['Y_O2_X_C'] + (p14 + p17 + p20 + p25) * self.kinetic_parameters['Y_O2_X_P'] + (p15 + p18 + p21 + p26) * self.kinetic_parameters['Y_O2_X_L'] + (p22 + p27) * self.kinetic_parameters['Y_O2_X_H'] + (p23 + p28) * self.kinetic_parameters['Y_O2_X_LG']) - p34
        O2_cons_m = (p13 + p19) * self.kinetic_parameters['Y_O2_X_C'] + (p14 + p20) * self.kinetic_parameters['Y_O2_X_P'] + (p15 + p21) * self.kinetic_parameters['Y_O2_X_L'] + (p22) * self.kinetic_parameters['Y_O2_X_H']
        O2_cons_t = (p16 + p24) * self.kinetic_parameters['Y_O2_X_C'] + (p17 + p25) * self.kinetic_parameters['Y_O2_X_P'] + (p18 + p26) * self.kinetic_parameters['Y_O2_X_L'] + (p27) * self.kinetic_parameters['Y_O2_X_H']
        dCH4_dt = +((p13 + p16 + p19 + p24) * self.kinetic_parameters['Y_CO2_X_C'] + (p14 + p17 + p20 + p25) * self.kinetic_parameters['Y_CO2_X_P'] + (p15 + p18 + p21 + p26) * self.kinetic_parameters['Y_CO2_X_L'] + (p22 + p27) * self.kinetic_parameters['Y_CO2_X_H'] + (p23 + p28) * self.kinetic_parameters['Y_CO2_X_LG']) * ((CH4_emission_factor) / 100) * (16.04 / 44)
        # Update cumulative CH4 emissions
        self.cumulative_emissions['CH4'] += dCH4_dt * 0.001  # Convert to kg

        # Update compost production data
        self.compost_production['mass'] = TM
        self.compost_production['quality_parameters']['moisture_content'] = W / TM
        self.compost_production['quality_parameters']['organic_matter'] = (X_C + X_CE + X_H + X_LG + X_L) / TM
        self.compost_production['quality_parameters']['cn_ratio'] = self.calculate_CN_ratio(X_C, X_P, X_L, X_H, X_CE, X_LG)
        dCO2_dt = +((p13 + p16 + p19 + p24) * self.kinetic_parameters['Y_CO2_X_C'] + (p14 + p17 + p20 + p25) * self.kinetic_parameters['Y_CO2_X_P'] + (p15 + p18 + p21 + p26) * self.kinetic_parameters['Y_CO2_X_L'] + (p22 + p27) * self.kinetic_parameters['Y_CO2_X_H'] + (p23 + p28) * self.kinetic_parameters['Y_CO2_X_LG']) * ((100 - CH4_emission_factor) / 100)
        dCO2_atm_dt = p35
        # Update cumulative CO2 emissions
        self.cumulative_emissions['CO2'] += dCO2_dt * 0.001  # Convert to kg

        dNH3_dt = +((p13 + p16 + p19 + p24) * self.kinetic_parameters['Y_NH3_X_C'] + (p14 + p17 + p20 + p25) * self.kinetic_parameters['Y_NH3_X_P'] + (p15 + p18 + p21 + p26) * self.kinetic_parameters['Y_NH3_X_L'] + (p22 + p27) * self.kinetic_parameters['Y_NH3_X_H'] + (p23 + p28) * self.kinetic_parameters['Y_NH3_X_LG'])
        dNH3_atm_dt = p36
        # Update cumulative NH3 emissions
        self.cumulative_emissions['NH3'] += dNH3_dt * 0.001  # Convert to kg
        dH2O_dt = ((p13 + p16 + p19 + p24) * self.kinetic_parameters['Y_H2O_X_C'] + (p14 + p17 + p20 + p25) * self.kinetic_parameters['Y_H2O_X_P'] + (p15 + p18 + p21 + p26) * self.kinetic_parameters['Y_H2O_X_L'] + (p22 + p27) * self.kinetic_parameters['Y_H2O_X_H'] + (p23 + p28) * self.kinetic_parameters['Y_H2O_X_LG']) - p37
        dH20_atm_dt = p37
        dTM_dt = -(p35+p36+p37)
        masses = {
        "C": (X_C+S_C) / (X_C+S_C+X_P+S_P+X_L+S_L+X_H+S_H+X_CE+S_CE+X_LG+S_LG+X_A+X_In),
        "P": (X_P+S_P) / (X_C+S_C+X_P+S_P+X_L+S_L+X_H+S_H+X_CE+S_CE+X_LG+S_LG+X_A+X_In),
        "L": (X_L+S_L) / (X_C+S_C+X_P+S_P+X_L+S_L+X_H+S_H+X_CE+S_CE+X_LG+S_LG+X_A+X_In),
        "H": (X_H+S_H) / (X_C+S_C+X_P+S_P+X_L+S_L+X_H+S_H+X_CE+S_CE+X_LG+S_LG+X_A+X_In),
        "CE": (X_CE+S_CE) / (X_C+S_C+X_P+S_P+X_L+S_L+X_H+S_H+X_CE+S_CE+X_LG+S_LG+X_A+X_In),
        "LG": (X_LG+S_LG) / (X_C+S_C+X_P+S_P+X_L+S_L+X_H+S_H+X_CE+S_CE+X_LG+S_LG+X_A+X_In),
        "A": X_A / (X_C+S_C+X_P+S_P+X_L+S_L+X_H+S_H+X_CE+S_CE+X_LG+S_LG+X_A+X_In),
        "In":X_In/(X_C+S_C+X_P+S_P+X_L+S_L+X_H+S_H+X_CE+S_CE+X_LG+S_LG+X_A+X_In),
        }
        cp_i = [2037, 2018, 1200, 1200, 1350, 1137, 1300]
        mass_values = list(masses.values())

        mix_heat_capacity = 4186 * X_Mc + np.sum([cp * mass for cp, mass in zip(cp_i, mass_values)])

        q_bio_m = (1.3 * 10 ** 7) * O2_cons_m
        q_bio_t = (1.6 * 10 ** 7) * O2_cons_t
        q_bio = q_bio_m + q_bio_t
        q_evap = p37 * 2257000
        dT_dt = (q_bio - (q_conv + q_evap)) / (mix_heat_capacity * TM)

        return [
            dX_MB_dt, dX_TB_dt, dX_MA_F_dt, dX_TA_F_dt, dX_DB_dt, dW_dt, dX_P_dt, dX_L_dt, dX_C_dt, dX_H_dt,
            dX_CE_dt, dX_LG_dt, dX_A_dt, dX_In_dt, dS_C_dt, dS_P_dt, dS_L_dt, dS_H_dt, dS_CE_dt, dS_LG_dt,
            dTM_dt, dT_dt, dO2_dt, dCO2_dt, dCH4_dt, dNH3_dt, dH2O_dt, dCO2_atm_dt, dNH3_atm_dt, dH20_atm_dt,
            dC_C_dt, dC_P_dt, dC_L_dt, dC_H_dt, dC_CE_dt, dC_LG_dt
        ]
