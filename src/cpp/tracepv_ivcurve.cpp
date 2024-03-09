#include "../../include/cpp/tracepv_header.h" // TRACE-PV/include/cpp/

extern void IVcurve(const PVCell& pvCell, double T, double Ir, double V, double I, double* result) {
    const double T_plus_const = T + T_const;
    const double VT = k * T_plus_const * pvCell.DI_factor * pvCell.Ncell / q;
    const double I_ph = (pvCell.IL + pvCell.Temp_isc * (T_plus_const - T_ref)) * Ir / Ir_ref;
    const double exp_term = std::exp((V + pvCell.Rs * I) / VT);
    const double g = I_ph - pvCell.I0 * (exp_term - 1) - (V + pvCell.Rs * I) / pvCell.Rsh - I;
    const double g_d = -pvCell.I0 * exp_term * pvCell.Rs / VT - pvCell.Rs / pvCell.Rsh - 1;
    result[0] = g;
    result[1] = g_d;
}

extern vector<double> IVcurve_lookup(const PVCell& cell, double temp, double ir) {
    const double curr_Voc = cell.Voc + cell.Ncell * k * (temp + 273.15) * cell.DI_factor / q * (log(ir / Ir_ref)) + cell.Temp_soc * (temp + 273.15 - T_ref);
    const double h = curr_Voc / cell.iter;
    const double curr_Isc = (ir / Ir_ref) * cell.Isc * cell.Nstring;
    const int max_idx = floor(cell.Vmp / h / cell.Ncell);
    const double max_I = cell.Imp * cell.Nstring;

    vector<double> I(cell.iter, 0.0);
    I[0] = curr_Isc;
    I[max_idx] = max_I;

    for (int i = 1; i < cell.iter; ++i) {
        if (i == max_idx) continue;

        I[i] = I[i - 1];
        if (ir == 0.0) {
            I[i] = 0.0;
            continue;
        }

        double I_next = I[i];
        double g[2];
        const double V = i * h;
        do {
            g[0] = IVcurve(cell, temp, ir, V, I_next)[0];
            g[1] = IVcurve(cell, temp, ir, V, I_next)[1];
            I_next -= g[0] / g[1];
        } while (abs(I_next - I[i]) >= 1e-5);
        I[i] = I_next * cell.Nstring;
    }

    return I;
}
	

extern double* IVcurve_status(const PVCell& pvCell, double T, double Ir, double V_boost, double V_ref) {
    
    static constexpr size_t N = 1000; // Maximum number of iterations
    double status[2] = { 0.0, 0.0 };

    if (Ir == 0) {
        return status;
    }

    const double T1 = T + T_const;
    const double curr_Voc = pvCell.Voc + pvCell.Ncell * k * T1 * pvCell.DI_factor / q * log(Ir / Ir_ref) + pvCell.Temp_soc * (T1 - T_ref);
    const double h = curr_Voc / N;
    const double Nmodule_inv = 1.0 / pvCell.Nmodule;

    std::vector<double> I(N + 1, 0.0);
    std::vector<double> V(N + 1);

    I[0] = pvCell.Isc * Ir / Ir_ref;

    for (size_t i = 0; i <= N; ++i) {
        V[i] = i * h;
    }

    for (size_t i = 1; i < N; ++i) {
        I[i] = I[i - 1];
        while (true) {
            double g, g_d;
            // IVcurve function implementation is required here
            g = IVcurve(pvCell, T, Ir, V[i], I[i])[0];
            g_d = IVcurve(pvCell, T, Ir, V[i], I[i])[1];
            const double I_next = I[i] - g / g_d;
            if (std::abs(I_next - I[i]) < 1e-5) {
                I[i] = I_next;
                break;
            }
            I[i] = I_next;
        }
    }

    const size_t curr_idx_mpp = std::round(V_ref * Nmodule_inv / h);
    for (size_t i = 0; i < curr_idx_mpp; ++i) {
        const double power = V[i] * I[i] * pvCell.Nstring;
        if (power > status[0]) {
            status[0] = power;
        }
    }

    const size_t curr_idx_boost = std::round(V_boost * Nmodule_inv / h);
    status[1] = V_boost * I[curr_idx_boost] * pvCell.Nstring;

    return status;
}