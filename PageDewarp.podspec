Pod::Spec.new do |s|
  s.name         = 'PageDewarp'
  s.version      = '2.0.0'
  s.summary      = 'Page dewarping for document images'
  s.homepage     = 'https://github.com/erykpiast/page-dewarp-swift'
  s.license      = { type: 'MIT' }
  s.authors      = { 'Eryk Napierala' => 'eryk@napierala.pl' }
  s.source       = { git: 'https://github.com/erykpiast/page-dewarp-swift.git', tag: s.version }

  s.ios.deployment_target = '16.0'
  s.swift_version = '5.9'
  s.static_framework = true
  s.libraries = 'c++'
  s.frameworks = 'UIKit', 'Accelerate'

  s.dependency 'opencv-rne', '~> 4.11'

  s.subspec 'CLBFGSB' do |c|
    c.source_files = 'Sources/CLBFGSB/**/*.{c,h}'
    c.public_header_files = 'Sources/CLBFGSB/include/*.h'
    c.header_dir = 'CLBFGSB'
  end

  s.subspec 'OpenCVBridge' do |b|
    b.source_files = 'Sources/OpenCVBridge/**/*.{h,m,mm}'
    b.public_header_files = 'Sources/OpenCVBridge/include/*.h'
    b.pod_target_xcconfig = {
      'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    }
  end

  s.subspec 'Core' do |core|
    core.source_files = 'Sources/PageDewarp/**/*.swift'
    core.dependency 'PageDewarp/OpenCVBridge'
    core.dependency 'PageDewarp/CLBFGSB'
  end

  s.default_subspecs = 'Core'
end
