package { [
    'build-essential',
    'vim',
    'curl',
    'language-pack-pt',
    'htop',
    'tmux',
    'python3-venv',
    'python3-pip',
    'python3.6'
  ]:
  ensure  => 'installed',
}

exec { 'install python packages':
   command   => 'pip3 install -r /src/requirements/requirements.txt',
   path => '/usr/bin'
}