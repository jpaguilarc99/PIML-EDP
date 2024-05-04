import matplotlib.pyplot as plt

def plot_results(losses, L2_errors, h1_norms):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Graficar la función de pérdida 
    ax.semilogy(losses, label="Función de pérdida")

    # Graficar el error L2 
    ax.semilogy(L2_errors, label="Norma $L^2$")       
    ax.semilogy(h1_norms, label="Norma $H^1$") 

    # Configurar los títulos y etiquetas    
    ax.set_xlabel("Iteración", fontsize="x-large")  
    ax.set_ylabel('Error relativo', fontsize="x-large") 
    ax.legend(fontsize="large")

    # Ajustar el layout y mostrar la figura
    plt.tight_layout()
    plt.savefig('combined_semilog_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def style_plot(x, y, x_data, y_data, yh, i, xp=None):
    style = "clean.mplstyle"
    with plt.style.context(style):        
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.plot(x, y, label="Exact solution")
        if y_data is not None:            
            plt.scatter(x_data, y_data, label='Training data')
        if yh is not None:
            plt.scatter(x, yh, label="Neural network prediction", s=10, color='#377eb8')
        if xp is not None:
            plt.scatter(xp, -0*torch.ones_like(xp), label='Physics loss training locations', marker='.', s=6)
        plt.annotate("Training step: %i"%(i+1), 
             xy=(0, 1), 
             xycoords='axes fraction', 
             fontsize="small", 
             color="k",
             ha='left', 
             va='top')
        plt.legend(loc='best')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x)$')