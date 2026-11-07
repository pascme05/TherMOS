#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         DataLoader
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This class defines all loading operations for the data. 1D data are tables with one column per signal (time, id,
inputs, outputs, references) and are read from .xlsx/.csv files or from .mat files that store a struct of columns
named 'data'. 2D/3D data are .mat files with the arrays X (Nt x N), y (Nt x N), r (Nt x N), t, Ts and the geometry
and material information (geo, dx, dy, dz, Lx, Ly, Lz, k, rho, Cp, hc, fl, Ta). FE mesh objects (pde.FEMesh) cannot
be read by Python and are skipped.

V1.0: 07.11.2026, Pascal Schirmer
- Python port of DataLoader.m
- Time vector of ID based data always starts at 0 (F-D10), reference matrix sized by the references (F-D11)
- Ratio split without overlap and with validation data taken out of the training data (F-D9, legacy: MATLAB split)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================

# ==============================================================================
# External
# ==============================================================================
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import sys


#######################################################################################################################
# Class
#######################################################################################################################
class DataLoader:
    ###################################################################################################################
    # Constructor
    ###################################################################################################################
    def __init__(self, Ts, Type, Dim, Split, Intp, Inp, Out, path, legacy=0):
        self.Ts = Ts  # sampling time (sec)
        self.Type = Type  # input data type ('mat', 'xlsx', 'csv')
        self.Dim = Dim  # data dimension (1, 2, 3)
        self.Split = Split  # splitting method (1) file, (2) sheet, (3) ID, (4) ratio
        self.Intp = Intp  # interpolation method
        self.Inp = list(Inp)  # list of input features
        self.Out = list(Out)  # list of output features
        self.path = path  # path dictionary
        self.legacy = legacy  # (0): corrected behaviour, (1): MATLAB behaviour

    ###################################################################################################################
    # General Loading
    ###################################################################################################################
    def loadData(self, fileName, sheetName, stat, setup):
        # ==============================================================================
        # Description
        # ==============================================================================
        """
        This function loads one data file and returns a dictionary with t, X, y, r, Ts and the raw data.

        Input:
        1) fileName:    name of the data file without extension
        2) sheetName:   name of the sheet (xlsx only)
        3) stat:        (1) training, (2) testing, (3) validation
        4) setup:       includes all simulation variables

        Output:
        1) out:         loaded data
        """

        # ==============================================================================
        # Calculation
        # ==============================================================================
        if self.Type in ('xlsx', 'csv'):
            out = self._loadTable(fileName, sheetName, stat, setup)
        elif self.Type == 'mat' and self.Dim == 1:
            out = self._loadMat1D(fileName, stat, setup)
        elif self.Type == 'mat' and self.Dim >= 2:
            out = self._loadMat2D(fileName)
        else:
            sys.exit('ERROR: Wrong data type or dimension')

        # ==============================================================================
        # Return
        # ==============================================================================
        out.update({'Type': self.Type, 'Dim': self.Dim, 'Split': self.Split, 'Intp': self.Intp,
                    'Inp': self.Inp, 'Out': self.Out})
        return out

    ###################################################################################################################
    # Helper Functions
    ###################################################################################################################
    # ==============================================================================
    # Find File
    # ==============================================================================
    def _findFile(self, fileName, ext):
        name = str(fileName).replace('\\', '/').strip('/')
        direct = os.path.join(self.path['data'], *name.split('/')) + '.' + ext
        if os.path.isfile(direct):
            return direct
        base = os.path.basename(name) + '.' + ext
        for root, _, files in os.walk(self.path['data']):
            if base in files:
                return os.path.join(root, base)
        sys.exit('ERROR: Data file ' + base + ' not found under ' + self.path['data'])

    # ==============================================================================
    # Select Rows
    # ==============================================================================
    def _selectRows(self, df, stat, setup):
        # File and sheet based
        if self.Split in (1, 2):
            print('INFO: File and sheet based data')

        # ID based
        elif self.Split == 3:
            teID = list(np.atleast_1d(setup['teID']))
            vlID = list(np.atleast_1d(setup['vlID']))
            if stat == 1:
                ids = [i for i in np.unique(df['id']) if i not in teID and i not in vlID]
                if len(setup['trID']) > 0:
                    ids = [i for i in ids if i in list(setup['trID'])]
            elif stat == 2:
                ids = teID
            else:
                ids = vlID
            df = df[df['id'].isin(ids)].reset_index(drop=True)
            Ts_ID = df['time'].iloc[1] - df['time'].iloc[0]
            if self.legacy == 1 and self.Type in ('xlsx', 'csv'):
                df['time'] = np.linspace(df['time'].iloc[0], Ts_ID * len(df), len(df))
            else:
                df['time'] = np.arange(len(df)) * Ts_ID
            print('INFO: ID based data')

        # Ratio based
        else:
            N = len(df)
            trIdx = int(np.floor(N * setup['rTr']))
            if self.legacy == 1:
                if stat == 1:
                    df = df.iloc[0:trIdx]
                elif stat == 2:
                    df = df.iloc[trIdx - 1:]
                else:
                    df = df.iloc[0:int(np.floor(trIdx * setup['rVl']))]
            else:
                vlIdx = int(np.floor(trIdx * (1 - setup['rVl'])))
                if stat == 1:
                    df = df.iloc[0:vlIdx]
                elif stat == 2:
                    df = df.iloc[trIdx:]
                else:
                    df = df.iloc[vlIdx:trIdx]
            df = df.reset_index(drop=True)
            print('INFO: Split based data')

        return df

    # ==============================================================================
    # Select Input and Output
    # ==============================================================================
    def _selectInpOut(self, df, setup):
        for key in ['inp', 'out', 'ref']:
            miss = [c for c in setup[key] if c not in df.columns]
            if len(miss) > 0:
                sys.exit('ERROR: Columns ' + str(miss) + ' of setup["' + key + '"] do not exist in the data')

        out = {'t': df['time'].to_numpy(dtype=float),
               'X': df[list(setup['inp'])].to_numpy(dtype=float),
               'y': df[list(setup['out'])].to_numpy(dtype=float),
               'r': df[list(setup['ref'])].to_numpy(dtype=float),
               'Ts': self.Ts,
               'Data': df.drop(columns=[c for c in ['time', 'id'] if c in df.columns])}

        return out

    ###################################################################################################################
    # Loading Functions
    ###################################################################################################################
    # ==============================================================================
    # Table Data (xlsx/csv)
    # ==============================================================================
    def _loadTable(self, fileName, sheetName, stat, setup):
        try:
            if self.Type == 'csv':
                df = pd.read_csv(self._findFile(fileName, 'csv'))
            else:
                df = pd.read_excel(self._findFile(fileName, 'xlsx'), sheet_name=str(sheetName))
            print('INFO: Table data successfully loaded (' + str(fileName) + ', ' + str(sheetName) + ')')
        except SystemExit:
            raise
        except Exception as e:
            sys.exit('ERROR: Failed to load data from the specified worksheet: ' + str(e))

        # rows below the header that are not numeric (e.g. units as in MATLAB VariableUnits) are removed
        num = df.apply(pd.to_numeric, errors='coerce')
        nUnit = 0
        while nUnit < len(df) and num.iloc[nUnit].isna().all() and df.iloc[nUnit].notna().any():
            nUnit = nUnit + 1
        if nUnit > 0:
            print('INFO: ' + str(nUnit) + ' unit row(s) removed below the table header')
            df = num.iloc[nUnit:].reset_index(drop=True)
        df = self._selectRows(df, stat, setup)
        return self._selectInpOut(df, setup)

    # ==============================================================================
    # Matlab 1D Data
    # ==============================================================================
    def _loadMat1D(self, fileName, stat, setup):
        try:
            raw = sio.loadmat(self._findFile(fileName, 'mat'), squeeze_me=True, struct_as_record=False)['data']
            df = pd.DataFrame({f: np.atleast_1d(getattr(raw, f)) for f in raw._fieldnames})
            print('INFO: 1D Matlab data successfully loaded (' + str(fileName) + ')')
        except SystemExit:
            raise
        except Exception as e:
            sys.exit('ERROR: Failed to load 1D Matlab data (struct of columns named "data" expected): ' + str(e))

        df = self._selectRows(df, stat, setup)
        return self._selectInpOut(df, setup)

    # ==============================================================================
    # Matlab 2D/3D Data
    # ==============================================================================
    def _loadMat2D(self, fileName):
        keys = ['X', 'y', 'r', 't', 'Ts', 'geo', 'dx', 'dy', 'dz', 'Lx', 'Ly', 'Lz', 'k', 'rho', 'Cp', 'hc', 'fl', 'Ta']
        try:
            raw = sio.loadmat(self._findFile(fileName, 'mat'), variable_names=keys)
            print('INFO: ' + str(self.Dim) + 'D Matlab data successfully loaded (' + str(fileName) + ')')
        except SystemExit:
            raise
        except Exception as e:
            sys.exit('ERROR: Failed to load Matlab data: ' + str(e))

        # Reformat
        Data = {}
        for k in keys:
            if k not in raw:
                continue
            v = np.asarray(raw[k], dtype=float)
            if v.size == 1:
                Data[k] = float(v.ravel()[0])
            elif k in ('X', 'y', 'r', 'geo'):
                Data[k] = np.atleast_2d(v)
            else:
                Data[k] = v.ravel(order='F')

        # Time vector consistent with the number of samples
        Nt = Data['y'].shape[0]
        t = Data['t']
        if len(t) != Nt:
            print('WARN: Time vector has ' + str(len(t)) + ' samples but data has ' + str(Nt) + ', time vector rebuilt')
            t = t[0] + np.arange(Nt) * Data['Ts']

        out = {'t': t, 'X': Data['X'], 'y': Data['y'], 'r': Data['r'], 'Ts': Data['Ts'], 'Data': Data}

        return out


#######################################################################################################################
# References
#######################################################################################################################
