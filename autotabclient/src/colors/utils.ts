export function saturate(color: string, saturation: number) {
	var col = hexToRgb(color);
	if (!col) return color;
	var hsl = rgbToHsl(col.r, col.g, col.b);

	hsl.s += saturation / 100;

	var out = hslToRgb(hsl.h, hsl.s, hsl.l);

	return rgbToHex(out.r, out.g, out.b);
}

export function changeHue(color: string, degree: number) {
	var col = hexToRgb(color);
	if (!col) return color;
	var hsl = rgbToHsl(col.r, col.g, col.b);

	hsl.h += degree / 360;

	var out = hslToRgb(hsl.h, hsl.s, hsl.l);

	return rgbToHex(out.r, out.g, out.b);
}

export function changeTemperature(color: string, temperature: number) {
	var col = hexToRgb(color);
	if (!col) return;
	var temp = temperature / 100;

	col.r = Math.round(col.r * temp);
	col.g = Math.round(col.g * temp);
	col.b = Math.round(col.b * temp);

	var out = rgbToHex(col.r, col.g, col.b);

	return out;
}

export function darken(color: string, darken: number) {
	var col = hexToRgb(color);
	if (!col) return color;
	var temp = darken / 100;

	col.r = Math.round(col.r * temp);
	col.g = Math.round(col.g * temp);
	col.b = Math.round(col.b * temp);

	var out = rgbToHex(col.r, col.g, col.b);

	return out;
}

export function tint(color: string, tint: number) {
	var col = hexToRgb(color);
	if (!col) return color;
	var temp = tint / 100;

	col.r = Math.round(col.r + (255 - col.r) * temp);
	col.g = Math.round(col.g + (255 - col.g) * temp);
	col.b = Math.round(col.b + (255 - col.b) * temp);

	var out = rgbToHex(col.r, col.g, col.b);

	return out;
}

function componentToHex(c: number) {
	var hex = c.toString(16);
	return hex.length == 1 ? "0" + hex : hex;
}

function rgbToHex(r: number, g: number, b: number) {
	return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
}

function rgbToHsl(r: number, g: number, b: number) {
	r /= 255;
	g /= 255;
	b /= 255;

	var max = Math.max(r, g, b),
		min = Math.min(r, g, b);
	var h = 0,
		s = 0,
		l = (max + min) / 2;

	if (max != min) {
		var d = max - min;
		s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

		switch (max) {
			case r:
				h = ((g - b) / d + 6) % 6;
				break;
			case g:
				h = (b - r) / d + 2;
				break;
			case b:
				h = (r - g) / d + 4;
				break;
		}

		h /= 6;
	}

	return { h: h, s: s, l: l };
}

function hslToRgb(h: number, s: number, l: number) {
	var r, g, b;

	if (s == 0) {
		r = g = b = l;
	} else {
		var hue2rgb = function hue2rgb(p: number, q: number, t: number) {
			if (t < 0) t += 1;
			if (t > 1) t -= 1;
			if (t < 1 / 6) return p + (q - p) * 6 * t;
			if (t < 1 / 2) return q;
			if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
			return p;
		};

		var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
		var p = 2 * l - q;

		r = hue2rgb(p, q, h + 1 / 3);
		g = hue2rgb(p, q, h);
		b = hue2rgb(p, q, h - 1 / 3);
	}

	return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
}

function hexToRgb(hex: string) {
	var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
	return result
		? {
				r: parseInt(result[1], 16),
				g: parseInt(result[2], 16),
				b: parseInt(result[3], 16),
		  }
		: null;
}
