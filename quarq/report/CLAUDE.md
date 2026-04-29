# report/ — HTML/PDF report generation layer

## Layer contract

- **charts.py**: Pure functions. Accept DataFrames/dicts, return `go.Figure`. No side effects, no disk I/O.
- **renderer.py**: Consumes figures + metrics, returns HTML string or writes PDF. No chart computation here.
- **template.html**: Jinja2. No external CDN. First chart embeds Plotly JS inline (`include_plotlyjs="inline"`), subsequent charts use `False`.

## chart builders

| Function | Inputs | Chart type |
|---|---|---|
| `correlation_heatmap(returns)` | multi-ticker returns DataFrame | `go.Heatmap` |
| `weight_treemap(weights, sleeve_map)` | flat weights dict + sleeve map | `go.Treemap` |
| `cumulative_returns_line(port, bench, label)` | two daily returns Series | two `go.Scatter` traces |
| `drawdown_area(returns)` | portfolio returns Series | `go.Scatter` fill=tozeroy |
| `rolling_sharpe(returns, rfr, window=63)` | portfolio returns + rfr | `go.Scatter` |

## Do not use plotly.express

`px` wrappers are broken for Heatmap and Treemap in Plotly 6. Use `go.*` only.

## PDF export

`render_pdf` requires Playwright (`quarq[full]`). If missing, raises `QuarqError` with install instructions.
