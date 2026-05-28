import { spawnSync } from 'node:child_process'

const env = {
  ...process.env,
  FLOWHAMSTER_RUN_GPU_SMOKE: '1',
}

const npmCli = process.env.npm_execpath
const command = npmCli ? process.execPath : (process.platform === 'win32' ? 'npm.cmd' : 'npm')
const args = npmCli
  ? [npmCli, 'run', 'test:run', '--', 'smoke/gpuTrainingSmoke.test.ts']
  : ['run', 'test:run', '--', 'smoke/gpuTrainingSmoke.test.ts']

const result = spawnSync(command, args, {
  env,
  stdio: 'inherit',
})

if (result.error) {
  console.error(result.error)
}

process.exit(result.status ?? 1)
