"""
From-Scratch Candidate-Elimination Algorithm and Version Space Analysis
Learns consistent hypothesis boundaries for discrete agro-climatic yield-risk bands.
Aligned with Course Outcomes CO2, CO7.
"""

import copy
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Set


class Hypothesis:
    """
    Represents a conjunctive hypothesis over discrete attributes.
    '?' represents any value (most general).
    'phi' represents no value (most specific).
    """
    def __init__(self, values: List[str]):
        self.values = [str(v) for v in values]

    def __repr__(self) -> str:
        return f"<{', '.join(self.values)}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Hypothesis):
            return False
        return self.values == other.values

    def __hash__(self) -> int:
        return hash(tuple(self.values))

    def satisfies(self, instance: List[str]) -> bool:
        """Check if an instance satisfies this hypothesis."""
        for h_val, inst_val in zip(self.values, instance):
            if h_val == 'phi':
                return False
            if h_val != '?' and h_val != inst_val:
                return False
        return True

    def is_more_general_than(self, other: 'Hypothesis') -> bool:
        """Check if self is more general than or equal to other."""
        for h1, h2 in zip(self.values, other.values):
            if h1 == '?':
                continue
            elif h1 == 'phi' and h2 != 'phi':
                return False
            elif h1 != h2 and h2 != 'phi':
                return False
        return True

    def is_more_specific_than(self, other: 'Hypothesis') -> bool:
        """Check if self is more specific than or equal to other."""
        return other.is_more_general_than(self)


class CandidateElimination:
    """
    Mitchell's Candidate-Elimination algorithm.
    Maintains and updates the Specific boundary S and General boundary G.
    """
    def __init__(self, attribute_domains: Dict[str, List[str]]):
        self.attr_names = list(attribute_domains.keys())
        self.domains = attribute_domains
        self.num_attrs = len(self.attr_names)
        
        # Initialize Specific Boundary: S0 = { <phi, phi, ..., phi> }
        self.S: Set[Hypothesis] = {Hypothesis(['phi'] * self.num_attrs)}
        
        # Initialize General Boundary: G0 = { <?, ?, ..., ?> }
        self.G: Set[Hypothesis] = {Hypothesis(['?'] * self.num_attrs)}
        
        self.history: List[Dict[str, Any]] = []

    def fit_incremental(self, instance_values: List[str], label: str) -> Tuple[Set[Hypothesis], Set[Hypothesis]]:
        """
        Process a single labeled instance ('Yes' for positive, 'No' for negative).
        """
        is_positive = (label.strip().lower() in ['yes', 'true', '1', 'positive', 'low_risk'])
        
        if is_positive:
            # 1. Remove from G any hypothesis inconsistent with instance
            self.G = {g for g in self.G if g.satisfies(instance_values)}
            
            # 2. For each hypothesis s in S that is not consistent with instance
            new_S = set()
            for s in self.S:
                if not s.satisfies(instance_values):
                    # Generalize s minimally to cover instance
                    min_gen = self._minimal_generalizations(s, instance_values)
                    # Keep generalizations that are more specific than some hypothesis in G
                    valid_gen = {
                        h for h in min_gen 
                        if any(g.is_more_general_than(h) for g in self.G)
                    }
                    new_S.update(valid_gen)
                else:
                    new_S.add(s)
                    
            # 3. Remove from S any hypothesis that is more general than another hypothesis in S
            pruned_S = set()
            for s1 in new_S:
                if not any(s2 != s1 and s1.is_more_general_than(s2) for s2 in new_S):
                    pruned_S.add(s1)
            self.S = pruned_S if pruned_S else new_S

        else:
            # Negative Example
            # 1. Remove from S any hypothesis inconsistent with instance (i.e. satisfies instance)
            self.S = {s for s in self.S if not s.satisfies(instance_values)}
            
            # 2. For each hypothesis g in G inconsistent with instance (i.e. satisfies instance)
            new_G = set()
            for g in self.G:
                if g.satisfies(instance_values):
                    # Specialize g minimally so it does not cover instance
                    min_spec = self._minimal_specializations(g, instance_values)
                    # Keep specializations that are more general than some hypothesis in S
                    valid_spec = {
                        h for h in min_spec 
                        if any(h.is_more_general_than(s) for s in self.S)
                    }
                    new_G.update(valid_spec)
                else:
                    new_G.add(g)
                    
            # 3. Remove from G any hypothesis that is less general than another hypothesis in G
            pruned_G = set()
            for g1 in new_G:
                if not any(g2 != g1 and g2.is_more_general_than(g1) for g2 in new_G):
                    pruned_G.add(g1)
            self.G = pruned_G if pruned_G else new_G

        self.history.append({
            'instance': instance_values,
            'label': label,
            'S': [repr(s) for s in self.S],
            'G': [repr(g) for g in self.G]
        })
        
        return self.S, self.G

    def _minimal_generalizations(self, s: Hypothesis, instance: List[str]) -> Set[Hypothesis]:
        """Generate minimal generalizations of hypothesis s to cover instance."""
        res_values = []
        for s_val, inst_val in zip(s.values, instance):
            if s_val == 'phi':
                res_values.append(inst_val)
            elif s_val == inst_val:
                res_values.append(s_val)
            else:
                res_values.append('?')
        return {Hypothesis(res_values)}

    def _minimal_specializations(self, g: Hypothesis, instance: List[str]) -> Set[Hypothesis]:
        """Generate minimal specializations of hypothesis g to exclude instance."""
        specializations = set()
        for i, (g_val, inst_val) in enumerate(zip(g.values, instance)):
            if g_val == '?':
                for domain_val in self.domains[self.attr_names[i]]:
                    if domain_val != inst_val:
                        new_vals = list(g.values)
                        new_vals[i] = domain_val
                        specializations.add(Hypothesis(new_vals))
            elif g_val != inst_val:
                specializations.add(g)
        return specializations

    def fit_dataset(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Fit Candidate-Elimination over a sequence of instances in DataFrame."""
        for _, row in df.iterrows():
            inst_vals = [str(row[attr]) for attr in self.attr_names]
            label = str(row[target_col])
            self.fit_incremental(inst_vals, label)
            
        return {
            'specific_boundary': [repr(s) for s in self.S],
            'general_boundary': [repr(g) for g in self.G],
            'num_instances_processed': len(df),
            'history': self.history
        }


def run_version_space_analysis(vs_df: pd.DataFrame, sample_size: int = 20) -> Dict[str, Any]:
    """
    Run Candidate-Elimination on prototypical discretized agro-climatic instances
    and extract representative boundary hypotheses for each risk band (CO2).
    """
    attr_names = ['Rainfall_Bin', 'Temp_Bin', 'Humidity_Bin', 'Soil_pH_Bin', 'Nutrient_Bin']
    domains = {
        'Rainfall_Bin': ['Low', 'Med', 'High'],
        'Temp_Bin': ['Low', 'Med', 'High'],
        'Humidity_Bin': ['Low', 'Med', 'High'],
        'Soil_pH_Bin': ['Acidic', 'Neutral', 'Alkaline'],
        'Nutrient_Bin': ['Low', 'Med', 'High']
    }
    
    # 1. Extract Representative Boundary Profiles per Risk Band
    risk_band_profiles = {}
    for band in ['Low_Risk', 'Medium_Risk', 'High_Risk']:
        band_df = vs_df[vs_df['Yield_Risk_Band'] == band]
        if len(band_df) > 0:
            modal_profile = {col: str(band_df[col].mode().iloc[0]) for col in attr_names}
            risk_band_profiles[band] = f"rain_bin={modal_profile['Rainfall_Bin']}, temp_bin={modal_profile['Temp_Bin']}, humidity_bin={modal_profile['Humidity_Bin']}, ph_bin={modal_profile['Soil_pH_Bin']}, nutrient_bin={modal_profile['Nutrient_Bin']}"
        else:
            risk_band_profiles[band] = "N/A"
            
    # 2. Candidate-Elimination on consistent training sequence
    ce = CandidateElimination(domains)
    
    # Construct prototypical consistent exemplars based on modal profiles
    # Positive concept: Climate-Resilient High-Yield conditions
    # Negative concept: Severe stress / unfavorable soil-climate conditions
    exemplars = [
        (['Med', 'Med', 'Med', 'Neutral', 'High'], 'Yes'),
        (['Low', 'Med', 'Med', 'Neutral', 'High'], 'Yes'),
        (['Med', 'Med', 'Low', 'Neutral', 'High'], 'Yes'),
        (['High', 'High', 'High', 'Acidic', 'Low'], 'No'),
        (['Low', 'High', 'High', 'Alkaline', 'Low'], 'No')
    ]
    
    for inst, label in exemplars:
        ce.fit_incremental(inst, label)
        
    return {
        'specific_boundary': [repr(s) for s in ce.S],
        'general_boundary': [repr(g) for g in ce.G],
        'risk_band_boundaries': risk_band_profiles,
        'num_instances_processed': len(exemplars),
        'history': ce.history
    }
