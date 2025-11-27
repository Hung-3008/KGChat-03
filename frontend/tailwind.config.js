/** @type {import('tailwindcss').Config} */
export default {
    darkMode: 'class',
    content: [
        "./index.html",
        "./*.{js,ts,jsx,tsx}",
        "./components/**/*.{js,ts,jsx,tsx}",
        "./services/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: {
                    DEFAULT: '#008080', // Medical Teal
                    dark: '#006666',
                    light: '#339999',
                },
                secondary: {
                    DEFAULT: '#0F4C75', // Trustworthy Deep Blue
                    light: '#3282B8',
                },
                accent: {
                    amber: '#FFBF00', // Alert Amber
                    coral: '#FF7F50', // Soft Coral
                },
                background: {
                    light: '#F8F9FA', // Off-white/Cool Gray
                    dark: '#1A202C',
                },
            },
            fontFamily: {
                display: ['Inter', 'system-ui', 'sans-serif'],
                body: ['Inter', 'system-ui', 'sans-serif'],
            },
        },
    },
    plugins: [],
}
