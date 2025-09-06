import type {NextConfig} from 'next';

const nextConfig: NextConfig = {
  /* config options here */
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  webpack: (config, { isServer }) => {
    // Handle missing optional dependencies gracefully
    config.resolve.fallback = {
      ...config.resolve.fallback,
      '@genkit-ai/firebase': false,
      '@opentelemetry/exporter-jaeger': false,
    };
    
    // Ignore module not found warnings for optional dependencies
    config.externals = config.externals || [];
    if (isServer) {
      config.externals.push('@genkit-ai/firebase');
    }
    
    return config;
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'placehold.co',
        port: '',
        pathname: '/**',
      },
    ],
  },
};

export default nextConfig;
