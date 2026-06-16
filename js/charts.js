/* ============================================================
   APEX — Chart.js Visualizations
   ============================================================ */

const ApexCharts = {
  instances: {},

  // ── Shared Defaults ─────────────────────────────────────────
  defaults: {
    font: { family: "'Inter', sans-serif", size: 11 },
    color: '#6b7fa3',
    grid: {
      color: 'rgba(30,45,64,0.6)',
      borderDash: [4, 4],
    }
  },

  destroy(id) {
    if (this.instances[id]) {
      this.instances[id].destroy();
      delete this.instances[id];
    }
  },

  // ── Equity Curve ─────────────────────────────────────────────
  equityCurve(canvasId, data) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const isPositive = (data[data.length - 1] ?? 0) >= (data[0] ?? 0);
    const color = isPositive ? '#00e676' : '#ff3d71';

    const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
    gradient.addColorStop(0, isPositive ? 'rgba(0,230,118,0.25)' : 'rgba(255,61,113,0.25)');
    gradient.addColorStop(1, 'rgba(0,0,0,0)');

    this.instances[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map((_, i) => `Day ${i + 1}`),
        datasets: [{
          label: 'Equity',
          data,
          borderColor: color,
          borderWidth: 2,
          backgroundColor: gradient,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: color,
          pointHoverBorderColor: '#fff',
          pointHoverBorderWidth: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 800, easing: 'easeInOutQuart' },
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#111827',
            borderColor: '#1e2d40',
            borderWidth: 1,
            titleColor: '#e8edf5',
            bodyColor: '#6b7fa3',
            titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
            bodyFont:  { family: "'JetBrains Mono', monospace", size: 11 },
            callbacks: {
              label: ctx => `  $${ctx.parsed.y.toLocaleString()}`,
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#3d4f6b', font: { size: 10 }, maxTicksLimit: 8 },
            border: { display: false }
          },
          y: {
            grid: { color: 'rgba(30,45,64,0.5)', borderDash: [4,4] },
            ticks: {
              color: '#6b7fa3', font: { size: 10 },
              callback: v => `$${(v/1000).toFixed(0)}k`
            },
            border: { display: false }
          }
        }
      }
    });
  },

  // ── Drawdown Chart ───────────────────────────────────────────
  drawdownChart(canvasId, data) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    this.instances[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [{
          label: 'Drawdown %',
          data: data.values,
          borderColor: '#ff3d71',
          borderWidth: 2,
          backgroundColor: 'rgba(255,61,113,0.1)',
          fill: true,
          tension: 0.3,
          stepped: false,
          pointRadius: 0,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#3d4f6b', font: { size: 10 } }, border: { display: false } },
          y: {
            grid: { color: 'rgba(30,45,64,0.5)', borderDash: [4,4] },
            ticks: { color: '#6b7fa3', font: { size: 10 }, callback: v => `${v.toFixed(1)}%` },
            border: { display: false }
          }
        }
      }
    });
  },

  // ── Sector Donut ─────────────────────────────────────────────
  sectorDonut(canvasId, sectors) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const colors = [
      '#00d4ff','#7c3aed','#00e676','#ff3d71','#ffb300',
      '#448aff','#ff6b6b','#a78bfa','#34d399','#f97316',
      '#06b6d4','#8b5cf6','#10b981','#ef4444',
    ];

    this.instances[canvasId] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: sectors.map(s => s.name),
        datasets: [{
          data: sectors.map(s => s.exposure),
          backgroundColor: colors.slice(0, sectors.length).map(c => c + '99'),
          borderColor: colors.slice(0, sectors.length),
          borderWidth: 1,
          hoverOffset: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '70%',
        animation: { animateRotate: true, duration: 800 },
        plugins: {
          legend: {
            position: 'right',
            labels: {
              color: '#6b7fa3',
              font: { size: 11 },
              boxWidth: 10,
              padding: 12,
            }
          },
          tooltip: {
            backgroundColor: '#111827',
            borderColor: '#1e2d40',
            borderWidth: 1,
            titleColor: '#e8edf5',
            bodyColor: '#6b7fa3',
            callbacks: {
              label: ctx => `  ${ctx.parsed.toFixed(1)}%`
            }
          }
        }
      }
    });
  },

  // ── Win Rate Gauge (Radial) ──────────────────────────────────
  winRateGauge(canvasId, winRate) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const pct   = Math.min(Math.max(winRate, 0), 1);
    const color = pct >= 0.55 ? '#00e676' : pct >= 0.45 ? '#ffb300' : '#ff3d71';

    this.instances[canvasId] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [pct * 100, (1 - pct) * 100],
          backgroundColor: [color + 'cc', '#1e2d40'],
          borderWidth: 0,
          hoverOffset: 0,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '80%',
        rotation: -90,
        circumference: 180,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { animateRotate: true, duration: 1000 },
      },
      plugins: [{
        id: 'center-text',
        afterDraw(chart) {
          const { ctx, chartArea: { left, right, top, bottom } } = chart;
          const cx = (left + right) / 2;
          const cy = bottom - 10;
          ctx.save();
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = color;
          ctx.font = "bold 20px 'JetBrains Mono'";
          ctx.fillText(`${(pct * 100).toFixed(1)}%`, cx, cy);
          ctx.fillStyle = '#6b7fa3';
          ctx.font = "10px 'Inter'";
          ctx.fillText('WIN RATE', cx, cy + 18);
          ctx.restore();
        }
      }]
    });
  },

  // ── Conviction Bar Chart ─────────────────────────────────────
  convictionBars(canvasId, breakdown) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const colors = breakdown.map(b => {
      const frac = b.score / b.max;
      if (frac >= 0.9) return '#00d4ff';
      if (frac >= 0.5) return '#00e676';
      if (frac > 0)    return '#ffb300';
      return '#ff3d71';
    });

    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: breakdown.map(b => b.label),
        datasets: [{
          data: breakdown.map(b => b.score),
          backgroundColor: colors.map(c => c + '33'),
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 4,
          borderSkipped: false,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600 },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: {
            min: 0, max: 2,
            grid: { color: 'rgba(30,45,64,0.5)', borderDash: [4,4] },
            ticks: { color: '#6b7fa3', font: { size: 10 }, stepSize: 1 },
            border: { display: false }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#6b7fa3', font: { size: 10 } },
            border: { display: false }
          }
        }
      }
    });
  },

  // ── R-Multiple Distribution ──────────────────────────────────
  rMultipleDist(canvasId, trades) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const closed = trades.filter(t => t.status === 'CLOSED' && t.rMultiple != null);
    if (!closed.length) return;

    const colors = closed.map(t => t.rMultiple > 0 ? 'rgba(0,230,118,0.7)' : 'rgba(255,61,113,0.7)');
    const borders = closed.map(t => t.rMultiple > 0 ? '#00e676' : '#ff3d71');

    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: closed.map(t => t.ticker),
        datasets: [{
          data: closed.map(t => t.rMultiple),
          backgroundColor: colors,
          borderColor: borders,
          borderWidth: 1,
          borderRadius: 3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#6b7fa3', font: { size: 10 } },
            border: { display: false }
          },
          y: {
            grid: { color: 'rgba(30,45,64,0.5)', borderDash: [4,4] },
            ticks: { color: '#6b7fa3', font: { size: 10 }, callback: v => `${v}R` },
            border: { display: false }
          }
        }
      }
    });
  },
};
