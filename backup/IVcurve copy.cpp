#include "kernel.h"

extern vector<double> IVcurve(PVCell pvCell, double T, double Ir, double V, double I) {
    
    std::vector<double> result;
    
    T = T + T_const;
    double VT = k * T * pvCell.DI_factor * pvCell.Ncell / q;
    
    double I_ph = (pvCell.IL + pvCell.Temp_isc * (T - T_ref)) * Ir / Ir_ref;
    double g = I_ph - pvCell.I0 * (std::exp((V + pvCell.Rs * I) / VT) - 1) - (V + pvCell.Rs * I) / pvCell.Rsh - I;
    double g_d = -pvCell.I0 * std::exp((V + pvCell.Rs * I) / VT) * pvCell.Rs / VT - pvCell.Rs / pvCell.Rsh - 1;
    
    result.push_back(g);
    result.push_back(g_d);
    
    return result;
}


extern vector<double> IVcurve_lookup(PVCell cell, double temp, double ir){

    double curr_Voc=cell.Voc, curr_Isc=cell.Isc, h, I_next, V=0.0;
    int max_idx = 0;

    curr_Voc = cell.Voc + cell.Ncell*k*(temp+273.15)*cell.DI_factor/q*(log(ir/Ir_ref))+cell.Temp_soc*(temp+273.15-T_ref);
    
    h = curr_Voc/cell.iter;
    I_next = 0.0;
	
    vector<double> I(cell.iter,0.0);
	vector<double> g(2,0.0);
	curr_Isc = (ir/Ir_ref) * cell.Isc * cell.Nstring; //*(cell.Isc + cell.Temp_isc*(temp+273.15-temp_ref))*cell.Nstring;
    
    cout << "curr_Isc: " << curr_Isc << endl;
    
    I[0] = curr_Isc;
	
    max_idx = floor(cell.Vmp/h/cell.Ncell);

    I[max_idx] = cell.Imp*cell.Nstring;
    
 
	for(int i=1; i<cell.iter; ++i){
		I[i] = I[i-1];
		if(ir == 0.0){
	    		I[0] = 0.0;
			break;
	    	}
		if(i==max_idx)
			continue;
		while(1){
			g = IVcurve(cell, temp, ir, V+i*h, I[i]);
			I_next = I[i] - g[0]/g[1];
			if(abs(I_next-I[i])<1e-5){
				I[i] = I_next;
				break;
            }
			I[i] = I_next;
		}
		I[i]*=cell.Nstring;
	}

	return I;
}


extern double IVcurve_mpp(PVCell pvCell, double T, double Ir, double V_ref) {
    
    double T1 = T + T_const;
    double* I = new double[pvCell.iter]{ 0.0 };
    
    if (Ir == 0) {
        delete[] I;
        return 0.0;
    }

    double curr_Voc = pvCell.Voc + pvCell.Ncell * k * T1 * pvCell.DI_factor / q * log(Ir / Ir_ref) + pvCell.Temp_soc * (T1 - T_ref);
    double h = curr_Voc / pvCell.iter;
    double* V = new double[pvCell.iter + 1]{ 0.0 };

    for (int i = 0; i <= pvCell.iter; i++) {
        V[i] = i * h;
    }

    I[0] = pvCell.Isc * Ir / Ir_ref;

    for (int i = 1; i < pvCell.iter; i++) {
        I[i] = I[i - 1];
        while (true) {
            double g, g_d;
            // IVcurve function implementation is required here
            g = IVcurve(pvCell, T, Ir, V[i], I[i])[0];
            g_d = IVcurve(pvCell, T, Ir, V[i], I[i])[1];
        
            double I_next = I[i] - g / g_d;
            
            if (std::abs(I_next - I[i]) < 1e-5) {
                I[i] = I_next;
                break;
            }
            I[i] = I_next;   
        }
    }   
    
    for (int i = 0; i <= pvCell.iter; i++) {
        V[i] *= pvCell.Nmodule;
        I[i] *= pvCell.Nstring;
    }

    int curr_idx = std::round(V_ref / h / pvCell.Nmodule);
    
    double MPP = 0.0;

    for (int i = 0; i < curr_idx; i++) {
        double power = V[i] * I[i];
        if (power > MPP) {
            MPP = power;
        }
    }

    delete[] I;
    delete[] V;

    return MPP;
}

extern double IVcurve_boost(PVCell pvCell, double T, double Ir, double V_boost) {
    
    double T1 = T + T_const;
    double* I = new double[pvCell.iter]{ 0.0 };
    
    if (Ir == 0) {
        delete[] I;
        return 0.0;
    }

    double curr_Voc = pvCell.Voc + pvCell.Ncell * k * T1 * pvCell.DI_factor / q * log(Ir / Ir_ref) + pvCell.Temp_soc * (T1 - T_ref);
    double h = curr_Voc / pvCell.iter;
    double* V = new double[pvCell.iter + 1]{ 0.0 };

    for (int i = 0; i <= pvCell.iter; i++) {
        V[i] = i * h;
    }

    I[0] = pvCell.Isc * Ir / Ir_ref;

    for (int i = 1; i < pvCell.iter; i++) {
        I[i] = I[i - 1];
        while (true) {
            double g, g_d;
            // IVcurve function implementation is required here
            g = IVcurve(pvCell, T, Ir, V[i], I[i])[0];
            g_d = IVcurve(pvCell, T, Ir, V[i], I[i])[1];
            double I_next = I[i] - g / g_d;
            if (std::abs(I_next - I[i]) < 1e-5) {
                I[i] = I_next;
                break;
            }
            I[i] = I_next;
        }
    }

    for (int i = 0; i <= pvCell.iter; i++) {
        V[i] *= pvCell.Nmodule;
        I[i] *= pvCell.Nstring;
    }

    int curr_idx = std::round(V_boost / h / pvCell.Nmodule);

    double power = V_boost * I[curr_idx];

    delete[] I;
    delete[] V;

    return power;
}
