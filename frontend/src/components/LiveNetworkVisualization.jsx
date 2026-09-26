import { COLORS } from '../lib/theme';

const WIDTH = 640;
const HEIGHT = 260;

function nodePosition(index, total) {
  const angle = (index / Math.max(total, 1)) * Math.PI * 2 - Math.PI / 2;
  const rx = WIDTH / 2 - 46;
  const ry = HEIGHT / 2 - 34;
  return { x: WIDTH / 2 + rx * Math.cos(angle), y: HEIGHT / 2 + ry * Math.sin(angle) };
}

// A calm, scientific-looking flow diagram: events arrive from surrounding
// nodes toward a central hub, colored by the detector's real prediction for
// that real replayed record. Native SVG <animate>/<animateMotion> keeps this
// lightweight -- no animation loop or extra dependency.
export default function LiveNetworkVisualization({ events }) {
  const hub = { x: WIDTH / 2, y: HEIGHT / 2 };

  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="network-viz" preserveAspectRatio="xMidYMid meet">
      <circle cx={hub.x} cy={hub.y} r={20} fill={COLORS.primaryDark} />
      <circle cx={hub.x} cy={hub.y} r={20} fill="none" stroke={COLORS.accent} strokeWidth={1.4} opacity={0.4}>
        <animate attributeName="r" values="20;34;20" dur="3.2s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.5;0;0.5" dur="3.2s" repeatCount="indefinite" />
      </circle>

      {events.map((event, i) => {
        const pos = nodePosition(i, events.length);
        const isAttack = event.prediction === 'attack';
        const color = isAttack ? COLORS.threat : COLORS.success;
        const midX = (pos.x + hub.x) / 2;
        const midY = (pos.y + hub.y) / 2 - 18;
        const path = `M ${pos.x} ${pos.y} Q ${midX} ${midY} ${hub.x} ${hub.y}`;
        const duration = 2 + (i % 4) * 0.35;

        return (
          <g key={`${event.id}-${i}`}>
            <path d={path} fill="none" stroke={color} strokeWidth={1.1} opacity={0.28} />
            <circle r={3} fill={color}>
              <animateMotion dur={`${duration}s`} repeatCount="indefinite" path={path} />
            </circle>
            <circle cx={pos.x} cy={pos.y} r={4.5} fill={color} opacity={0.85} />
          </g>
        );
      })}
    </svg>
  );
}
