import matplotlib.pyplot as plt

def plot_results(losses, L2_errors, h1_norms):
    style = "clean.mplstyle"
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(10, 4))

        # Graficar la función de pérdida 
        ax.loglog(losses, label="Función de pérdida")

        # Graficar el error L2 
        ax.loglog(L2_errors, label="$Error_{L^{2}(\Omega)}$")       
        ax.loglog(h1_norms, label="$Error_{H^{1}(\Omega)}$") 

        # Configurar los títulos y etiquetas    
        ax.set_xlabel("Iteración", fontsize="x-large")       
        ax.legend(fontsize="large")

        # Ajustar el layout y mostrar la figura
        plt.tight_layout()
        plt.savefig('combined_loglog_plot.pdf', dpi=300, bbox_inches='tight')
        plt.show()

def style_plot(x, y, x_data, y_data, yh, i, xp=None):
    style = "clean.mplstyle"
    with plt.style.context(style):        
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.plot(x, y, label="Solución exacta")
        if y_data is not None:            
            plt.scatter(x_data, y_data, label='Datos de entrada')
        if yh is not None:
            plt.scatter(x, yh, label="Predicción red neuronal", s=10, color='#377eb8')
        if xp is not None:
            plt.scatter(xp, -0*torch.ones_like(xp), label='Physics loss training locations', marker='.', s=6)
        plt.annotate("Iteración: %i"%(i+1), 
             xy=(0, 1), 
             xycoords='axes fraction', 
             fontsize="small", 
             color="k",
             ha='left', 
             va='top')
        plt.legend(loc='best')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x)$')