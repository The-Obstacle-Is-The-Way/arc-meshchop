# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in ARC-MeshChop, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainers directly with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

We will respond within 7 days and work with you to address the issue.

## Security Considerations

### Data Handling

- This project processes medical imaging data (MRI)
- No patient data is stored or transmitted by the software itself
- Users are responsible for handling their data according to applicable regulations (HIPAA, GDPR, etc.)

### Dependencies

- Dependencies are managed via `uv` with lockfile (`uv.lock`)
- Regular dependency updates are performed
- No known vulnerable dependencies at time of release

### Model Security

- Trained model weights are stored as PyTorch checkpoints (`.pt` files)
- ONNX exports contain only model architecture and weights
- No embedded code execution in exported models

## Best Practices

When using ARC-MeshChop:

1. **Keep dependencies updated**: Run `uv sync` periodically
2. **Validate data sources**: Only use data from trusted sources
3. **Secure storage**: Protect any medical data according to regulations
4. **Review checkpoints**: Only load model checkpoints from trusted sources

## No Warranty

This software is provided "as is" without warranty of any kind. It is intended for research purposes and should not be used for clinical decision-making without proper validation.
