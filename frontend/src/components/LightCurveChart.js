import React from 'react';
import Plot from 'react-plotly.js';

const LightCurveChart = ({ lightCurveData, title = "Light Curve Analysis", showTransits = true }) => {
  if (!lightCurveData || !lightCurveData.time || !lightCurveData.flux) {
    return (
      <div className="bg-slate-900/70 backdrop-blur-sm rounded-xl p-8 border border-slate-800">
        <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
        <p className="text-slate-400">No light curve data available</p>
      </div>
    );
  }

  const { time, flux, flux_err, transit_period, transit_depth, data_source } = lightCurveData;

  // Create the main light curve trace
  const traces = [
    {
      x: time,
      y: flux,
      type: 'scatter',
      mode: 'markers',
      marker: {
        size: 3,
        color: '#60a5fa',
        opacity: 0.7
      },
      name: 'Observed Flux',
      hovertemplate: 'Time: %{x:.2f} days<br>Flux: %{y:.6f}<extra></extra>'
    }
  ];

  // Add error bars if available
  if (flux_err) {
    traces[0].error_y = {
      type: 'data',
      array: flux_err,
      visible: true,
      color: '#94a3b8'
    };
  }

  // Add transit markers if transit data is available
  if (showTransits && transit_period && transit_depth) {
    // Calculate transit times
    const maxTime = Math.max(...time);
    const transitTimes = [];
    for (let t = 0; t <= maxTime; t += transit_period) {
      transitTimes.push(t);
    }

    // Add vertical lines for predicted transits
    const transitLines = {
      x: transitTimes.flatMap(t => [t, t, null]),
      y: transitTimes.flatMap(() => [Math.min(...flux) * 0.999, Math.max(...flux) * 1.001, null]),
      type: 'scatter',
      mode: 'lines',
      line: {
        color: '#ef4444',
        width: 2,
        dash: 'dash'
      },
      name: 'Predicted Transits',
      hoverinfo: 'skip',
      showlegend: true
    };
    traces.push(transitLines);
  }

  const layout = {
    title: {
      text: title,
      font: { color: '#ffffff', size: 16 }
    },
    xaxis: {
      title: 'Time (days)',
      color: '#e2e8f0',
      gridcolor: '#475569',
      showgrid: true
    },
    yaxis: {
      title: 'Relative Flux',
      color: '#e2e8f0',
      gridcolor: '#475569',
      showgrid: true
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15, 23, 42, 0.8)',
    font: { color: '#e2e8f0' },
    legend: {
      font: { color: '#e2e8f0' },
      bgcolor: 'rgba(15, 23, 42, 0.8)'
    },
    margin: { l: 60, r: 30, t: 50, b: 60 },
    height: 400
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    displaylogo: false
  };

  return (
    <div className="bg-slate-900/70 backdrop-blur-sm rounded-xl p-6 border border-slate-800">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          📈 {title}
        </h3>
        {data_source && (
          <p className="text-slate-400 text-sm mt-1">Data Source: {data_source}</p>
        )}
        {transit_period && (
          <div className="flex gap-4 text-sm text-slate-300 mt-2">
            <span>Period: {transit_period.toFixed(2)} days</span>
            {transit_depth && (
              <span>Depth: {(transit_depth * 100).toFixed(3)}%</span>
            )}
          </div>
        )}
      </div>
      
      <Plot
        data={traces}
        layout={layout}
        config={config}
        style={{ width: '100%' }}
      />
    </div>
  );
};

export default LightCurveChart;