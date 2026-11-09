#pragma once

#include <functional>
#include <stdexcept>
#include <utility>

namespace metal4
{
// Keep this after the renderer resources that platform callbacks may borrow.
// Shutdown must retire those callbacks even after a partial bind.
class renderer_runtime_registration final
{
	std::function<void()> m_shutdown;

  public:
	renderer_runtime_registration() = default;
	renderer_runtime_registration(const renderer_runtime_registration&) = delete;
	renderer_runtime_registration& operator=(const renderer_runtime_registration&) = delete;
	~renderer_runtime_registration()
	{
		try { close(); }
		catch (...) {} // Explicit close reports failures on the normal exit path.
	}

	void install(const std::function<void()>& shutdown, const auto& bind)
	{
		if (m_shutdown || !shutdown)
			throw std::logic_error("Native renderer runtime registration requires an idle owner and shutdown callback");
		m_shutdown = shutdown;
		bind();
	}

	void close()
	{
		// A shutdown that reports an error may already have retired its hooks.
		if (auto shutdown = std::exchange(m_shutdown, {})) shutdown();
	}
};
} // namespace metal4
