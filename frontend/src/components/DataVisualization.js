import React from 'react';
import Plot from 'react-plotly.js';
import { TrendingUp, BarChart3, PieChart, Scatter3D } from 'lucide-react';

const DataVisualization = ({ 
  planetData, 
  comparisonData, 
  showCorrelations = true, 
  showDistributions = true 
}) => {
  if (!planetData) {
    return (
      <div className="bg-slate-900/70 backdrop-blur-sm rounded-xl p-8 border border-slate-800">
        <p className="text-slate-400">No visualization data available</p>
      </div>
    );
  }

  // Create mass vs radius plot
  const createMassRadiusPlot = () => {
    if (!comparisonData || !comparisonData.similar_planets) {
      return null;
    }

    const similar = comparisonData.similar_planets.filter(p => p.pl_masse && p.pl_rade);
    const target = comparisonData.target_planet;

    const traces = [
      {
        x: similar.map(p => p.pl_masse),
        y: similar.map(p => p.pl_rade),
        type: 'scatter',
        mode: 'markers',
        marker: {
          size: 8,
          color: '#60a5fa',
          opacity: 0.6
        },
        name: 'Similar Exoplanets',
        text: similar.map(p => p.pl_name),
        hovertemplate: '%{text}<br>Mass: %{x:.2f} M⊕<br>Radius: %{y:.2f} R⊕<extra></extra>'
      }
    ];

    // Add target planet
    if (target.mass && target.radius) {
      traces.push({
        x: [target.mass],
        y: [target.radius],
        type: 'scatter',
        mode: 'markers',
        marker: {
          size: 12,
          color: '#ef4444',
          symbol: 'star'
        },
        name: 'Target Planet',
        text: [target.name],
        hovertemplate: '%{text}<br>Mass: %{x:.2f} M⊕<br>Radius: %{y:.2f} R⊕<extra></extra>'
      });
    }

    return {
      data: traces,
      layout: {
        title: { text: 'Mass vs Radius Comparison', font: { color: '#ffffff' } },
        xaxis: { 
          title: 'Planet Mass (Earth masses)', 
          color: '#e2e8f0',
          gridcolor: '#475569',
          type: 'log'
        },
        yaxis: { 
          title: 'Planet Radius (Earth radii)', 
          color: '#e2e8f0',
          gridcolor: '#475569',
          type: 'log'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(15, 23, 42, 0.8)',
        font: { color: '#e2e8f0' },
        legend: { font: { color: '#e2e8f0' } },
        height: 400
      }
    };
  };

  // Create orbital period distribution
  const createPeriodDistribution = () => {
    if (!comparisonData || !comparisonData.similar_planets) {
      return null;
    }

    const periods = comparisonData.similar_planets
      .filter(p => p.pl_orbper)
      .map(p => p.pl_orbper);

    return {
      data: [{
        x: periods,
        type: 'histogram',
        marker: {
          color: '#34d399',
          opacity: 0.7
        },
        name: 'Orbital Periods'
      }],
      layout: {
        title: { text: 'Orbital Period Distribution', font: { color: '#ffffff' } },
        xaxis: { 
          title: 'Orbital Period (days)', 
          color: '#e2e8f0',
          gridcolor: '#475569'
        },
        yaxis: { 
          title: 'Count', 
          color: '#e2e8f0',
          gridcolor: '#475569'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(15, 23, 42, 0.8)',
        font: { color: '#e2e8f0' },
        height: 350
      }
    };
  };

  // Create temperature vs distance plot
  const createTempDistancePlot = () => {
    if (!comparisonData || !comparisonData.similar_planets) {
      return null;
    }

    const planets = comparisonData.similar_planets.filter(p => p.pl_eqt && p.pl_orbsmax);
    
    return {
      data: [{
        x: planets.map(p => p.pl_orbsmax),
        y: planets.map(p => p.pl_eqt),
        type: 'scatter',
        mode: 'markers',
        marker: {
          size: 8,
          color: planets.map(p => p.pl_eqt),
          colorscale: 'Viridis',
          showscale: true,
          colorbar: {
            title: 'Temperature (K)',
            titlefont: { color: '#e2e8f0' },
            tickfont: { color: '#e2e8f0' }
          }
        },
        text: planets.map(p => p.pl_name),
        hovertemplate: '%{text}<br>Distance: %{x:.3f} AU<br>Temperature: %{y:.0f} K<extra></extra>'
      }],
      layout: {
        title: { text: 'Temperature vs Orbital Distance', font: { color: '#ffffff' } },
        xaxis: { 
          title: 'Semi-major Axis (AU)', 
          color: '#e2e8f0',
          gridcolor: '#475569'
        },
        yaxis: { 
          title: 'Equilibrium Temperature (K)', 
          color: '#e2e8f0',
          gridcolor: '#475569'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(15, 23, 42, 0.8)',
        font: { color: '#e2e8f0' },
        height: 400
      }
    };
  };

  // Create planet type distribution pie chart
  const createTypeDistribution = () => {
    if (!comparisonData || !comparisonData.similar_planets) {
      return null;
    }

    // Classify planets by type based on radius
    const classifyPlanet = (radius) => {
      if (radius < 1.25) return 'Terrestrial';
      if (radius < 2.0) return 'Super-Earth';
      if (radius < 4.0) return 'Sub-Neptune';
      return 'Gas Giant';
    };

    const types = comparisonData.similar_planets
      .filter(p => p.pl_rade)
      .map(p => classifyPlanet(p.pl_rade));

    const typeCounts = types.reduce((acc, type) => {
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});

    return {
      data: [{
        labels: Object.keys(typeCounts),
        values: Object.values(typeCounts),
        type: 'pie',
        marker: {
          colors: ['#60a5fa', '#34d399', '#fbbf24', '#f87171']
        },
        textinfo: 'label+percent',
        textfont: { color: '#ffffff' }
      }],
      layout: {
        title: { text: 'Planet Type Distribution', font: { color: '#ffffff' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(15, 23, 42, 0.8)',
        font: { color: '#e2e8f0' },
        showlegend: true,
        legend: { font: { color: '#e2e8f0' } },
        height: 350
      }
    };
  };

  const massRadiusPlot = createMassRadiusPlot();
  const periodDist = createPeriodDistribution();
  const tempDistPlot = createTempDistancePlot();
  const typeDist = createTypeDistribution();

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    displaylogo: false
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-900/70 backdrop-blur-sm rounded-xl p-6 border border-slate-800">
        <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-cyan-400" />
          Data Visualization Dashboard
        </h3>

        {/* Mass vs Radius Plot */}
        {massRadiusPlot && (
          <div className="mb-8">
            <div className="flex items-center gap-2 mb-4">
              <Scatter3D className="w-5 h-5 text-blue-400" />
              <h4 className="text-lg font-semibold text-white">Mass-Radius Relationship</h4>
            </div>
            <Plot
              data={massRadiusPlot.data}
              layout={massRadiusPlot.layout}
              config={config}
              style={{ width: '100%' }}
            />
            <p className="text-slate-400 text-sm mt-2">
              Comparison of your target planet with similar exoplanets. The red star shows your target.
            </p>
          </div>
        )}

        {/* Two-column layout for smaller charts */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          {/* Period Distribution */}
          {periodDist && (
            <div>
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-5 h-5 text-green-400" />
                <h4 className="text-lg font-semibold text-white">Period Distribution</h4>
              </div>
              <Plot
                data={periodDist.data}
                layout={periodDist.layout}
                config={config}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Planet Type Distribution */}
          {typeDist && (
            <div>
              <div className="flex items-center gap-2 mb-4">
                <PieChart className="w-5 h-5 text-yellow-400" />
                <h4 className="text-lg font-semibold text-white">Planet Types</h4>
              </div>
              <Plot
                data={typeDist.data}
                layout={typeDist.layout}
                config={config}
                style={{ width: '100%' }}
              />
            </div>
          )}
        </div>

        {/* Temperature vs Distance Plot */}
        {tempDistPlot && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-purple-400" />
              <h4 className="text-lg font-semibold text-white">Temperature vs Orbital Distance</h4>
            </div>
            <Plot
              data={tempDistPlot.data}
              layout={tempDistPlot.layout}
              config={config}
              style={{ width: '100%' }}
            />
            <p className="text-slate-400 text-sm mt-2">
              Relationship between orbital distance and equilibrium temperature for similar exoplanets.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataVisualization;