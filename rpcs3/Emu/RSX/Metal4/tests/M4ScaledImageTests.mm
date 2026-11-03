#include "M4GuardedGuestMemory.h"
#include "M4ScaledImage.h"
#include "M4ScaledImageMapping.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>

namespace
{
using namespace metal4;
void require(bool v, const char* message)
{
	if (!v)
		throw std::runtime_error(message);
}
class fixture_access final : public exclusive_guest_memory, public exclusive_guest_memory::view, public std::enable_shared_from_this<fixture_access>
{
	std::mutex mutex;
	std::array<std::byte, 65536> bytes{};

  public:
	unsigned notifications = 0;
	// Fixture storage never reallocates;the provider owns every physical bank.
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& op) override
	{
		if (auto* current = guest_memory_epoch::current(*this))
		{
			op(*current);
			return;
		}
		std::lock_guard lock(mutex);
		guest_memory_epoch epoch(*this, *this);
		op(*this);
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		const auto offset = range.address % bytes.size();
		if (range.address >= 2 * bytes.size() || range.size > bytes.size() - offset)
			throw std::out_of_range("Scaled-image fixture mapping invalid");
		return bytes.data() + offset;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override
	{
		return overlaps(guest_surface_range{a.address % 65536, a.size}, guest_surface_range{b.address % 65536, b.size});
	}
	void written(guest_surface_range) noexcept override { ++notifications; }
};
std::size_t offset(const guest_surface_layout& layout, unsigned x, unsigned y)
{
	if (!layout.morton)
		return y * layout.row_pitch + x * layout.pixel_bytes;
	x += layout.morton->origin[0];
	y += layout.morton->origin[1];
	std::size_t word = 0;
	unsigned shift = 0;
	for (unsigned bit = 1; bit < layout.morton->storage[0] || bit < layout.morton->storage[1]; bit *= 2)
	{
		if (bit < layout.morton->storage[0])
		{
			word |= std::size_t(bool(x & bit)) << shift;
			++shift;
		}
		if (bit < layout.morton->storage[1])
		{
			word |= std::size_t(bool(y & bit)) << shift;
			++shift;
		}
	}
	return word * layout.pixel_bytes;
}
using rgba = std::array<double, 4>;
rgba read(const std::vector<std::byte>& bytes, const guest_surface_layout& layout, int x, int y)
{
	x = std::clamp(x, 0, int(layout.width) - 1);
	y = std::clamp(y, 0, int(layout.height) - 1);
	const auto at = offset(layout, x, y);
	const auto u = [&](unsigned i)
	{ return std::to_integer<unsigned>(bytes[at + i]); };
	if (layout.pixel_bytes == 4)
		return {u(1) / 255., u(2) / 255., u(3) / 255., u(0) / 255.};
	const auto word = u(0) * 256 + u(1);
	return {((word >> 11) & 31) / 31., ((word >> 5) & 63) / 63., (word & 31) / 31., 1};
}
std::vector<std::byte> reference(const scaled_image_plan& p, const std::vector<std::byte>& source, std::vector<std::byte> target)
{
	for (unsigned y = 0; y < p.destination.height; ++y)
		for (unsigned x = 0; x < p.destination.width; ++x)
		{
			const auto to = offset(p.destination, x, y);
			rgba color;
			if (!p.interpolate)
			{
				const auto coord = [](unsigned i, unsigned src, unsigned dst, bool flip)
				{const auto denominator=std::uint64_t(dst)*2;auto numerator=(std::uint64_t(i)*2+1)*src;if(flip)numerator=std::uint64_t(src)*denominator-numerator;return unsigned(numerator/denominator); };
				const auto a = std::min<unsigned>(coord(x + p.clip_x, p.input_width, p.scaled_width, p.flip_x), p.source.width - 1);
				const auto b = std::min<unsigned>(coord(y + p.clip_y, p.input_height, p.scaled_height, p.flip_y), p.source.height - 1);
				if (p.source_format == p.destination_format)
				{
					std::copy_n(source.begin() + offset(p.source, a, b), p.source.pixel_bytes, target.begin() + to);
					continue;
				}
				color = read(source, p.source, a, b);
			}
			else
			{
				double u = (x + p.clip_x + 0.5) * p.input_width / p.scaled_width, v = (y + p.clip_y + 0.5) * p.input_height / p.scaled_height;
				if (p.flip_x)
					u = p.input_width - u;
				if (p.flip_y)
					v = p.input_height - v;
				u -= 0.5;
				v -= 0.5;
				const int a = std::floor(u), b = std::floor(v);
				const double dx = u - a, dy = v - b;
				color = {};
				for (unsigned yy = 0; yy < 2; ++yy)
					for (unsigned xx = 0; xx < 2; ++xx)
					{
						const auto texel = read(source, p.source, a + xx, b + yy);
						const auto weight = (xx ? dx : 1 - dx) * (yy ? dy : 1 - dy);
						for (unsigned c = 0; c < 4; ++c)
							color[c] += texel[c] * weight;
					}
			}
			if (p.destination.pixel_bytes == 4)
			{
				target[to] = std::byte(std::lround(color[3] * 255));
				for (unsigned c = 0; c < 3; ++c)
					target[to + 1 + c] = std::byte(std::lround(color[c] * 255));
			}
			else
			{
				const auto word = (unsigned(std::lround(color[0] * 31)) << 11) | (unsigned(std::lround(color[1] * 63)) << 5) | unsigned(std::lround(color[2] * 31));
				target[to] = std::byte(word >> 8);
				target[to + 1] = std::byte(word);
			}
		}
	return target;
}
class native_access final : public exclusive_guest_memory,public exclusive_guest_memory::view,public std::enable_shared_from_this<native_access>
{
	std::mutex mutex;
	std::vector<std::byte> bytes=std::vector<std::byte>(1u<<20,std::byte{0xcd});
  public:
	unsigned notifications=0;
	void execute(const std::function<void(view&)>& work) override
	{
		if(auto* active=guest_memory_epoch::current(*this)){work(*active);return;}
		std::lock_guard lock(mutex);guest_memory_epoch epoch(*this,*this);work(*this);
	}
	std::byte* map(guest_surface_range range,bool) override
	{
		const auto offset=range.address%bytes.size();if(range.address>=2*bytes.size()||range.size>bytes.size()-offset)throw std::out_of_range("Native scale fixture range");return bytes.data()+offset;
	}
	bool same_storage(guest_surface_range a,guest_surface_range b) override {return overlaps(guest_surface_range{a.address%unsigned(bytes.size()),a.size},guest_surface_range{b.address%unsigned(bytes.size()),b.size});}
	std::optional<guest_surface_range> rebase_storage(guest_surface_range source,guest_surface_range request) override
	{
		const auto a=source.address%unsigned(bytes.size()),b=request.address%unsigned(bytes.size());if(b<a||request.size>source.size||b-a>source.size-request.size)return {};return guest_surface_range{source.address+b-a,request.size};
	}
	bool equivalent_storage(guest_surface_range a,guest_surface_range b) override {return a.address%bytes.size()==b.address%bytes.size()&&a.size==b.size;}
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override {return shared_from_this();}
	void written(guest_surface_range) noexcept override {++notifications;}
	std::vector<std::byte> copy() {std::lock_guard lock(mutex);return bytes;}
};
class native_memory final : public guest_surface_memory
{
  public:
	std::shared_ptr<native_access> access=std::make_shared<native_access>();
	std::shared_ptr<guarded_guest_memory> raw=std::make_shared<guarded_guest_memory>(access);
	bool forbid_pixels=false;unsigned reads=0,writes=0;
	guest_surface_layout watch_surface(const guest_surface_layout& coverage) override {return raw->watch_surface(coverage);}
	guest_surface_snapshot read_surface(const guest_surface_layout& coverage) override {require(!forbid_pixels,"Native scale read VMpixels");++reads;return raw->read_surface(coverage);}
	std::uint64_t surface_revision(const guest_surface_layout& coverage) override {return raw->surface_revision(coverage);}
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range r) override {return raw->storage_identity(r);}
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range> ranges) override {return raw->retain_backing(ranges);}
	bool test_storage(const storage_test& test) override {return raw->test_storage(test);}
	bool test_rebase(const rebase_test& test) override {return raw->test_rebase(test);}
	std::shared_ptr<const physical_surface> capture_physical_surface(const guest_surface_layout& c,std::uint64_t v) override {return raw->capture_physical_surface(c,v);}
	bool physical_surface_current(const physical_surface& p) override {return raw->physical_surface_current(p);}
	std::optional<bool> physical_surface_overlaps_current(const physical_surface& p,guest_surface_range r) override {return raw->physical_surface_overlaps_current(p,r);}
	std::optional<bool> physical_identity_overlaps_current(const physical_identity& p,const guest_surface_layout& r) override {return raw->physical_identity_overlaps_current(p,r);}
	std::optional<bool> physical_surface_virtual_current(const physical_surface& p,std::uint64_t v) override {return raw->physical_surface_virtual_current(p,v);}
	std::optional<guest_surface_snapshot> read_physical_surface(const physical_surface& p) override {require(!forbid_pixels,"Native scale read physical VMpixels");++reads;return raw->read_physical_surface(p);}
	std::optional<physical_surface_publication> compare_write_physical(std::span<const physical_surface_write> w,std::span<const std::uint64_t> v) override {require(!forbid_pixels,"Native scale wrote physical VMpixels");++writes;return raw->compare_write_physical(w,v);}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout& c,std::uint64_t v,std::span<const std::byte> b) override {require(!forbid_pixels,"Native scale wrote VMpixels");++writes;return raw->compare_write(c,v,b);}
	std::optional<std::uint64_t> compare_write_many(std::span<const guest_surface_write> w) override {require(!forbid_pixels,"Native scale scattered VMpixels");++writes;return raw->compare_write_many(w);}
	void write(guest_surface_range r,std::span<const std::byte> b) override {require(!forbid_pixels,"Native scale stored VMpixels");++writes;raw->write(r,b);}
};
void native_scaled_image_cases()
{
	context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);sample_transfer samples(ctx);unsigned profiles=0;std::size_t bytes_checked=0;
	for(unsigned in:{2u,4u})for(unsigned out:{2u,4u})for(unsigned layout=0;layout<7;++layout)for(unsigned count:{1u,2u,4u})for(unsigned flags=0;flags<8;++flags)
	{
		@autoreleasepool
		{
			auto memory=std::make_shared<native_memory>();const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned address,unsigned,std::size_t){return address;},1u<<20,ledger,64,{capture,io});scaled_image_transfer transfer(ctx,surfaces,textures,memory,capture,1u<<20);
			const bool banked=layout>=4,sm=layout==1||layout==3||layout==6,dm=layout==2||layout==3||layout==6;const unsigned pitch=layout==5?768:256;
			const auto ssw=sm?std::optional<swizzle_region>{{{16,16,1},{2,3,0},{8,8,1}}}:std::nullopt,dsw=dm?std::optional<swizzle_region>{{{16,16,1},{2,3,0},{8,8,1}}}:std::nullopt;
			const auto st=banked?std::optional<banked_tile_layout>{{0x10000,0x40000,pitch,1,0x1203}}:std::nullopt,dt=banked?std::optional<banked_tile_layout>{{0x80000,0x40000,pitch,3,0x1405}}:std::nullopt;
			const unsigned sa=banked?st->address+st->offset:0x1000,da=banked?dt->address+dt->offset:0x6000;const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);
			const auto sf=in==2?surface_color::rgb565:surface_color::argb8,df=out==2?surface_color::rgb565:surface_color::argb8;
			const auto source=surfaces.acquire({sa,{8/factors.x,8/factors.y,count,sf,false},sm?0:banked?pitch:8*in+1,ssw,st});const auto destination=surfaces.acquire({da,{8/factors.x,8/factors.y,count,df,false},dm?0:banked?pitch:8*out+1,dsw,dt});
			std::vector<std::byte> input(64*in),background(64*out,std::byte{0xa5});for(unsigned i=0;i<input.size();++i)input[i]=std::byte((i*37+i/5+73)&255);
			const auto seed=[&](const cached_surface_ref& surface,surface_color format,const std::vector<std::byte>& data)
			{
				surfaces.prepare(surface,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto images)
				{
					id<MTLTexture> target=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:images[0].pixelFormat width:8 height:8 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;target=make_texture(commands,d);}texture_region r;r.size=MTLSizeMake(8,8,1);colors.upload(commands,format,make_upload(commands,data),{},target,r);if(count>1)samples.import(commands,target,images[0]);
				});
			};seed(source,sf,input);seed(destination,df,background);
			auto sg=plan_guest_surface(sa,3,2,in,sm?0:banked?pitch:8*in+1,sm?std::optional<swizzle_region>{{{16,16,1},{3,5,0},{3,2,1}}}:std::nullopt);
			auto dg=plan_guest_surface(da,4,3,out,dm?0:banked?pitch:8*out+1,dm?std::optional<swizzle_region>{{{16,16,1},{4,4,0},{4,3,1}}}:std::nullopt);
			auto sti=st,dti=dt;if(banked){if(!sm)sti->offset+=2*pitch+in;if(!dm)dti->offset+=pitch+2*out;sg.range.address=sti->address+sti->offset;dg.range.address=dti->address+dti->offset;sg=attach_banked_tile(sg,*sti);dg=attach_banked_tile(dg,*dti);}else{if(!sm)sg.range.address+=2*sg.row_pitch+in;if(!dm)dg.range.address+=dg.row_pitch+2*out;}
			scaled_image_plan plan{sg,dg,in==2?scaled_image_format::rgb565:scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,5,4,7,5,1,1,bool(flags&1),bool(flags&2),bool(flags&4),sti,dti};
			const auto packed_source=plan_guest_surface(0,3,2,in),packed_destination=plan_guest_surface(0,4,3,out);std::vector<std::byte> cropped(6*in),old(12*out);for(unsigned y=0;y<2;++y)for(unsigned x=0;x<3;++x)std::copy_n(input.begin()+((y+2)*8+x+1)*in,in,cropped.begin()+(y*3+x)*in);for(unsigned y=0;y<3;++y)for(unsigned x=0;x<4;++x)std::copy_n(background.begin()+((y+1)*8+x+2)*out,out,old.begin()+(y*4+x)*out);auto canonical=plan;canonical.source=packed_source;canonical.destination=packed_destination;canonical.source_tile.reset();canonical.destination_tile.reset();const auto patch=reference(canonical,cropped,old);auto expected=background;for(unsigned y=0;y<3;++y)for(unsigned x=0;x<4;++x)std::copy_n(patch.begin()+(y*4+x)*out,out,expected.begin()+((y+1)*8+x+2)*out);
			const auto vm=memory->access->copy();const auto reads=memory->reads,writes=memory->writes,notifications=memory->access->notifications;const auto before=surfaces.transaction_statistics();memory->forbid_pixels=true;transfer.execute(plan);memory->forbid_pixels=false;require(transfer.native_completed()==1&&transfer.vm_completed()==0&&!transfer.has_pending(),"Suitable scaled image bypassed native route");require(memory->reads==reads&&memory->writes==writes&&memory->access->notifications==notifications&&memory->access->copy()==vm,"Native scale read/wrote guest allocation");require(surfaces.status(source).dirty&&surfaces.status(destination).dirty&&surfaces.transaction_statistics().submitted==before.submitted+1,"Native scale lost authority oraddedsource submissions");
			const auto raw=make_buffer(ctx,expected.size(),buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{destination,false}},[&](context& commands,auto images)
			{
				id<MTLTexture> image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:images[0].pixelFormat width:8 height:8 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;image=make_texture(commands,d);samples.expand(commands,images[0],image);}texture_region r;r.size=MTLSizeMake(8,8,1);colors.download(commands,df,image,r,raw,{});
			});if(std::memcmp(raw.contents,expected.data(),expected.size()))throw std::runtime_error("Native scale bytes differ profile="+std::to_string(profiles)+" input="+std::to_string(in)+" output="+std::to_string(out)+" layout="+std::to_string(layout)+" samples="+std::to_string(count)+" flags="+std::to_string(flags));bytes_checked+=expected.size();++profiles;
		}
	}
	std::cout<<"Native scaled images:"<<profiles<<" RGB565/ARGB8 format/layout/sample/flip/filter profiles,"<<bytes_checked<<" exactguest-word/native-background bytes;cropped oddpitch/Morton/banked256-768 storage,1/2/4 individual samples,one native write submission,zero VMpixel reads/writes/notifications andunchanged wholeVM passed\n";
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			native_scaled_image_cases();
			context ctx{1};
			auto provider = std::make_shared<fixture_access>();
			auto memory = std::make_shared<guarded_guest_memory>(provider);
			auto budget = std::make_shared<heap_budget>(ctx, 64 << 20);
			ctx.set_buffer_budget(budget);
			surface_cache surfaces(ctx, memory, 64 << 20, budget);
			guest_texture_cache textures(ctx, memory, [](unsigned address, unsigned, std::size_t)
										 { return address; }, 1 << 20, budget);
			unsigned captures = 0;
			const cpu_capture_dispatch capture = [&](const auto& op)
			{++captures;provider->execute([&](auto&){op();}); };
			scaled_image_transfer transfer(ctx, surfaces, textures, memory, capture, 1 << 20);
			unsigned cases = 0;
			for (unsigned in : {2u, 4u})
				for (unsigned out : {2u, 4u})
					for (unsigned flags = 0; flags < 8; ++flags)
						for (unsigned morton = 0; morton < 4; ++morton)
						{
							const auto src = plan_guest_surface(0x1000, 3, 2, in, (morton & 1) ? 0 : in * 5,
																(morton & 1) ? std::optional<swizzle_region>{{{8, 4, 1}, {1, 1, 0}, {3, 2, 1}}} : std::nullopt);
							const auto dst = plan_guest_surface(0x4000, 4, 3, out, (morton & 2) ? 0 : out * 7,
																(morton & 2) ? std::optional<swizzle_region>{{{8, 8, 1}, {2, 3, 0}, {4, 3, 1}}} : std::nullopt);
							const scaled_image_plan plan{src, dst, in == 2 ? scaled_image_format::rgb565 : scaled_image_format::argb8,
														 out == 2 ? scaled_image_format::rgb565 : scaled_image_format::argb8, 3, 2, 7, 5, 1, 1, bool(flags & 1), bool(flags & 2), bool(flags & 4)};
							std::vector<std::byte> input(src.range.size), output(dst.range.size, std::byte{0xa5});
							for (unsigned i = 0; i < input.size(); ++i)
								input[i] = std::byte((i * 37 + 73) % 256);
							memory->write(src.range, input);
							memory->write(dst.range, output);
							const auto expected = reference(plan, input, output);
							const auto before = captures;
							transfer.execute(plan);
							const auto actual = memory->read_surface(dst).bytes;
							if (actual != expected)
								throw std::runtime_error("Native scaled-image pixel/padding differs at case " + std::to_string(cases));
							require(captures == before + 2 && !transfer.has_pending(), "Transfer lost separate CPU capture/commit epochs");
							++cases;
						}
			scaled_image_compute compute(ctx, 1 << 20);
			for (unsigned width = 1; width <= 17; ++width)
				for (unsigned out = 1; out <= 17; ++out)
					for (bool flip : {false, true})
					{
						const auto src = plan_guest_surface(0x1000, width, 1, 4), dst = plan_guest_surface(0x4000, out, 1, 4);
						const scaled_image_plan boundary{src, dst, scaled_image_format::argb8, scaled_image_format::argb8, width, 1, out, 1, 0, 0, flip};
						std::vector<std::byte> bytes(src.range.size);
						for (unsigned i = 0; i < bytes.size(); ++i)
							bytes[i] = std::byte(i);
						const auto input_buffer = make_upload(ctx, bytes), output_buffer = make_buffer(ctx, dst.range.size, buffer_storage::shared);
						ctx.begin();
						compute.encode(ctx, boundary, input_buffer, output_buffer);
						ctx.submit().wait();
						const auto actual = map_buffer(output_buffer, 0, dst.range.size);
						const auto expected = reference(boundary, bytes, std::vector<std::byte>(dst.range.size));
						require(std::equal(actual.begin(), actual.end(), expected.begin()), "Scaled nearest rational boundary differs");
					}
			const auto packed = plan_guest_surface(0x1000, 65536, 1, 2), expanded = plan_guest_surface(0x40000, 65536, 1, 4);
			std::vector<std::byte> words565(packed.range.size);
			for (unsigned i = 0; i < 65536; ++i)
			{
				words565[i * 2] = std::byte(i >> 8);
				words565[i * 2 + 1] = std::byte(i);
			}
			const auto raw = make_upload(ctx, words565), rgba_buffer = make_buffer(ctx, expanded.range.size, buffer_storage::shared), roundtrip = make_buffer(ctx, packed.range.size, buffer_storage::shared);
			ctx.begin();
			compute.encode(ctx, {packed, expanded, scaled_image_format::rgb565, scaled_image_format::argb8, 65536, 1, 65536, 1}, raw, rgba_buffer);
			ctx.submit().wait();
			const auto colors = map_buffer(rgba_buffer, 0, expanded.range.size);
			for (unsigned i = 0; i < 65536; ++i)
			{
				const std::array expected{255u, unsigned(std::lround(((i >> 11) & 31) * 255. / 31)), unsigned(std::lround(((i >> 5) & 63) * 255. / 63)), unsigned(std::lround((i & 31) * 255. / 31))};
				for (unsigned c = 0; c < 4; ++c)
					require(colors[i * 4 + c] == std::byte(expected[c]), "RGB565 full-domain expansion differs");
			}
			ctx.begin();
			compute.encode(ctx, {expanded, packed, scaled_image_format::argb8, scaled_image_format::rgb565, 65536, 1, 65536, 1}, rgba_buffer, roundtrip);
			ctx.submit().wait();
			const auto restored = map_buffer(roundtrip, 0, packed.range.size);
			require(std::equal(restored.begin(), restored.end(), words565.begin()), "RGB565 full-domain roundtrip differs");
			const auto source = plan_guest_surface(0x1000, 4, 2, 4, 16), target = plan_guest_surface(0x4000, 2, 2, 4, 16);
			const scaled_image_plan plan{source, target, scaled_image_format::argb8, scaled_image_format::argb8, 4, 2, 2, 2};
			std::vector<std::byte> input(source.range.size), output(target.range.size, std::byte{0xa5});
			for (unsigned i = 0; i < input.size(); ++i)
				input[i] = std::byte(i + 1);
			memory->write(source.range, input);
			memory->write(target.range, output);
			const auto expected = reference(plan, input, output);
			const auto gate = [ctx.device() newSharedEvent];
			struct release_gate
			{
				id<MTLSharedEvent> event;
				std::uint64_t value = 10;
				~release_gate() { event.signaledValue = value; }
			} cleanup{gate};
			const auto start_pending = [&](std::uint64_t value)
			{
				bool timed_out = false;
				try
				{
					transfer.execute(plan, std::chrono::milliseconds(1), [&](context& commands)
									 { commands.wait_event(gate, value); });
				}
				catch (const std::runtime_error&)
				{
					timed_out = true;
				}
				require(timed_out && transfer.has_pending(), "Transfer timeout did not retain GPU transaction");
				require(memory->read_surface(target).bytes == output, "Pending transfer published destination early");
			};
			start_pending(1);
			memory->write({target.range.address + 8, 1}, std::array{std::byte{0x5c}});
			memory->write({source.range.address, 1}, std::array{std::byte{0xee}});
			gate.signaledValue = 1;
			transfer.synchronize();
			auto preserved = expected;
			preserved[8] = std::byte{0x5c};
			require(memory->read_surface(target).bytes == preserved, "Snapshot source or concurrent target padding was overwritten");
			memory->write(source.range, input);
			memory->write(target.range, output);
			start_pending(2);
			memory->write({target.range.address, 1}, std::array{std::byte{0x55}});
			gate.signaledValue = 2;
			bool conflict = false;
			try
			{
				transfer.synchronize();
			}
			catch (const texture_revision_conflict&)
			{
				conflict = true;
			}
			auto changed = output;
			changed[0] = std::byte{0x55};
			require(conflict && !transfer.has_pending() && memory->read_surface(target).bytes == changed, "Conflicting target store was overwritten");
			transfer.execute(plan);
			require(memory->read_surface(target).bytes == expected, "Transfer retry failed after target conflict");
			bool encode_failed = false;
			memory->write(target.range, output);
			try
			{
				transfer.execute(plan, std::chrono::seconds(30), [](auto&)
								 { throw std::range_error("dependency refusal"); });
			}
			catch (const std::range_error&)
			{
				encode_failed = true;
			}
			require(encode_failed && !transfer.has_pending() && memory->read_surface(target).bytes == output, "Encoding failure published VM data");
			transfer.execute(plan);
			const auto before_invalid_timeout = captures;
			bool invalid_timeout = false;
			try
			{
				transfer.execute(plan, std::chrono::milliseconds(-1));
			}
			catch (const std::invalid_argument&)
			{
				invalid_timeout = true;
			}
			require(invalid_timeout && captures == before_invalid_timeout, "Invalid wait timeout entered CPU capture");
			// Cached immutable reads of the target must be invalidated while
			// independently retained old images preserve their original pixels.
			const auto cached_target = plan_guest_surface(0x6000, 2, 2, 4, 16);
			memory->write(cached_target.range, std::vector<std::byte>(cached_target.range.size));
			fragment_texture_registers words{};
			words.offset = 0x6000;
			words.format = (1u << 16) | (0xa5u << 8) | (2u << 4) | 2u | 8u;
			words.control0 = 1u << 31;
			words.control1 = 0xaae4;
			words.image_rect = (2u << 16) | 2u;
			words.control3 = (1u << 20) | 16u;
			const auto old = textures.resolve(words);
			const auto invalidations = textures.statistics().invalidations;
			const auto native_target = surfaces.acquire({0x6000, {2, 2, 1, surface_color::argb8, false}, 16, {}});
			surfaces.prepare(native_target);
			transfer.execute({source, cached_target, scaled_image_format::argb8, scaled_image_format::argb8, 4, 2, 2, 2});
			require(textures.statistics().invalidations > invalidations && !surfaces.status(native_target).initialized,
					"Native scaled-image failed to invalidate texture/surface target mirrors");
			const auto retained = make_buffer(ctx, 16, buffer_storage::shared);
			ctx.begin();
			texture_to_buffer(ctx, old.texture, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(2, 2, 1)}, retained, 0);
			ctx.submit().wait();
			for (const auto byte : map_buffer(retained, 0, 16))
				require(byte == std::byte{0}, "Scaled-image mutated a retained old native texture");
			// Source/destination physical mirrors overlap:must use the captured
			// source version,as for memmove,even though GPU buffers are disjoint.
			const auto overlap_src = plan_guest_surface(0x2000, 4, 1, 4, 16), overlap_dst = plan_guest_surface(0x12004, 4, 1, 4, 16);
			std::vector<std::byte> overlap(20);
			for (unsigned i = 0; i < 20; ++i)
				overlap[i] = std::byte(i + 17);
			memory->write({0x2000, 20}, overlap);
			transfer.execute({overlap_src, overlap_dst, scaled_image_format::argb8, scaled_image_format::argb8, 4, 1, 4, 1});
			auto moved = overlap;
			std::copy_n(overlap.begin(), 16, moved.begin() + 4);
			require(memory->read({0x2000, 20}).bytes == moved, "Physical overlap did not preserve captured source");
			// GPU-dirty guest storage must flush before the common CPU epoch.
			const auto dirty = surfaces.acquire({0x3000, {4, 1, 1, surface_color::argb8, false}, 16, {}});
			surfaces.clear(dirty, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 1});
			transfer.execute({plan_guest_surface(0x3000, 4, 1, 4), plan_guest_surface(0x5000, 4, 1, 4), scaled_image_format::argb8, scaled_image_format::argb8, 4, 1, 4, 1});
			const auto dirty_pixels = memory->read({0x5000, 16}).bytes;
			for (unsigned i = 0; i < 16; ++i)
				require(dirty_pixels[i] == std::byte(std::array{255u, 18u, 52u, 86u}[i % 4]), "Scaled transfer missed dirty native source");
			// Destroy a pending transfer:GPU resources remain completion-owned,
			// but no callback may publish destination after its owner disappears.
			memory->write(target.range, output);
			{
				scaled_image_transfer abandoned(ctx, surfaces, textures, memory, capture, 1 << 20);
				bool timed_out = false;
				try
				{
					abandoned.execute(plan, std::chrono::milliseconds(1), [&](auto& commands)
									  { commands.wait_event(gate, 3); });
				}
				catch (const std::runtime_error&)
				{
					timed_out = true;
				}
				require(timed_out && abandoned.has_pending(), "Abandon fixture lacks pending work");
			}
			gate.signaledValue = 3;
			ctx.wait_idle();
			require(memory->read_surface(target).bytes == output, "Destroyed transfer published late VM data");
			// Metadata has an independent lifetime from VM byte revisions. Real
			// pending work must reject a changed IO/tile assignment at publication.
			const scaled_image_mapping_epoch::mappings original_mappings{{{source.range.address, source.range.size, scaled_image_tile_mapping{0x10000, 65536, 768, 2, 1, 0x20000, 0, 0}},
																		  {target.range.address, target.range.size, scaled_image_tile_mapping{0x30000, 65536, 768, 1, 1, 0x40000, 0, 0}}}};
			const auto metadata_gate = [ctx.device() newSharedEvent];
			release_gate metadata_cleanup{metadata_gate, 1000};
			unsigned metadata_checks = 0;
			const auto validator = [&](const auto& current)
			{
				return scaled_image_mapping_epoch([current, &metadata_checks, provider]
												  {
					require(guest_memory_epoch::current(*provider) != nullptr, "Metadata validated outside CPU exclusion");
					++metadata_checks;
					return *current; });
			};
			// Capturing a stale mapping fails before any upload/publication.
			{
				const auto dirty_target = surfaces.acquire({target.range.address, {2, 2, 1, surface_color::argb8, false}, 16, {}});
				surfaces.clear(dirty_target, 0xf0, 0xff765432, 0, 255, {0, 0, 2, 2});
				auto current = std::make_shared<scaled_image_mapping_epoch::mappings>(original_mappings);
				std::optional<scaled_image_mapping_epoch> epoch;
				capture([&]
						{ epoch.emplace(validator(current)); });
				(*current)[1].tile.reset();
				const auto notifications = provider->notifications;
				bool refused = false;
				try
				{
					transfer.execute(plan, std::chrono::seconds(30), {}, [owned = std::move(*epoch)]
									 { owned.validate(); });
				}
				catch (const scaled_image_mapping_conflict&)
				{
					refused = true;
				}
				require(refused && !transfer.has_pending() && provider->notifications == notifications && surfaces.status(dirty_target).dirty && memory->read_surface(target).bytes == output,
						"Stale mapping entered publication");
				// Deliberate cleanup with the original mapping restored;the refused
				// transfer itself must not flush this dirty native image to VM.
				surfaces.prepare_cpu_read(target.range);
			}
			unsigned mapping_cases = 0;
			for (unsigned side = 0; side < 2; ++side)
				for (unsigned field = 0; field < 11; ++field)
				{
					memory->write(target.range, output);
					auto current = std::make_shared<scaled_image_mapping_epoch::mappings>(original_mappings);
					const std::weak_ptr<scaled_image_mapping_epoch::mappings> lifetime = current;
					const auto notifications = provider->notifications;
					const auto signal = ++mapping_cases;
					{
						std::optional<scaled_image_mapping_epoch> epoch;
						capture([&]
								{ epoch.emplace(validator(current)); });
						bool timed_out = false;
						try
						{
							transfer.execute(plan, std::chrono::milliseconds(1), [&](auto& commands)
											 { commands.wait_event(metadata_gate, signal); }, [owned = std::move(*epoch)]
											 { owned.validate(); });
						}
						catch (const std::runtime_error&)
						{
							timed_out = true;
						}
						require(timed_out && transfer.has_pending(), "Metadata fixture did not retain pending GPU work");
					}
					auto& mapping = (*current)[side];
					switch (field)
					{
					case 0:
						mapping.address += 65536;
						break;
					case 1:
						++mapping.size;
						break;
					case 2:
						mapping.tile->address += 65536;
						break;
					case 3:
						mapping.tile->size *= 2;
						break;
					case 4:
						mapping.tile->pitch *= 2;
						break;
					case 5:
						mapping.tile->bank ^= 1;
						break;
					case 6:
						mapping.tile->location ^= 1;
						break;
					case 7:
						mapping.tile->io_offset += 65536;
						break;
					case 8:
						++mapping.tile->compression;
						break;
					case 9:
						++mapping.tile->compression_base;
						break;
					case 10:
						mapping.tile.reset();
						break;
					}
					current.reset();
					require(!lifetime.expired(), "Pending transfer lost owned metadata query");
					metadata_gate.signaledValue = signal;
					bool refused = false;
					try
					{
						transfer.synchronize();
					}
					catch (const scaled_image_mapping_conflict&)
					{
						refused = true;
					}
					require(refused && !transfer.has_pending() && lifetime.expired() && provider->notifications == notifications &&
								memory->read_surface(target).bytes == output,
							"Changed mapping published pixels or leaked its query owner");
				}
			memory->write(source.range, input);
			memory->write(target.range, output);
			{
				auto current = std::make_shared<scaled_image_mapping_epoch::mappings>(original_mappings);
				const std::weak_ptr<scaled_image_mapping_epoch::mappings> lifetime = current;
				{
					std::optional<scaled_image_mapping_epoch> epoch;
					capture([&]
							{ epoch.emplace(validator(current)); });
					bool timed_out = false;
					try
					{
						transfer.execute(plan, std::chrono::milliseconds(1), [&](auto& commands)
										 { commands.wait_event(metadata_gate, 23); }, [owned = std::move(*epoch)]
										 { owned.validate(); });
					}
					catch (const std::runtime_error&)
					{
						timed_out = true;
					}
					require(timed_out && transfer.has_pending(), "Stable metadata fixture lacks pending GPU work");
				}
				current.reset();
				require(!lifetime.expired(), "Stable pending query lost ownership");
				metadata_gate.signaledValue = 23;
				transfer.synchronize();
				require(lifetime.expired() && memory->read_surface(target).bytes == expected, "Stable metadata failed fresh publication");
			}
			require(metadata_checks == 95, "Metadata checks did not cover CPU epochs and cache preflights");
			// A failing final query also discards publication and releases its
			// owner;pending teardown must never invoke the validator later.
			for (bool abandon : {false, true})
			{
				memory->write(target.range, output);
				auto fail = std::make_shared<bool>(false);
				const std::weak_ptr<bool> lifetime = fail;
				unsigned checks = 0;
				const auto notifications = provider->notifications;
				const unsigned signal = abandon ? 25 : 24;
				{
					scaled_image_transfer isolated(ctx, surfaces, textures, memory, capture, 1 << 20);
					bool timed_out = false;
					try
					{
						isolated.execute(plan, std::chrono::milliseconds(1), [&](auto& commands)
										 { commands.wait_event(metadata_gate, signal); }, [fail, &checks, provider]
										 {
								require(guest_memory_epoch::current(*provider) != nullptr, "Query failure escaped CPU exclusion");
								++checks;
								if (*fail) throw std::range_error("mapping query failed"); });
					}
					catch (const std::runtime_error&)
					{
						timed_out = true;
					}
					require(timed_out && isolated.has_pending() && checks == 2, "Failing query fixture lacks pending work");
					*fail = true;
					fail.reset();
					require(!lifetime.expired(), "Pending failing query owner released early");
					if (!abandon)
					{
						metadata_gate.signaledValue = signal;
						bool refused = false;
						try
						{
							isolated.synchronize();
						}
						catch (const std::range_error&)
						{
							refused = true;
						}
						require(refused && !isolated.has_pending() && checks == 3, "Final query error did not discard publication");
					}
				}
				require(lifetime.expired(), "Retired query retained its owner");
				metadata_gate.signaledValue = signal;
				ctx.wait_idle();
				require(checks == (abandon ? 2 : 3) && provider->notifications == notifications && memory->read_surface(target).bytes == output,
						"Failed or destroyed metadata transaction published late VM data");
			}
			// Callbacks cannot replace the caller's plan after its bytes have been
			// captured. Both the encoded operation and published range stay owned.
			memory->write(source.range, input);
			memory->write(target.range, output);
			auto mutable_plan = plan;
			transfer.execute(mutable_plan, std::chrono::seconds(30), [&](auto&)
							 {
				mutable_plan.source.width = 1;
				mutable_plan.destination.range.address = 0x7000; });
			require(memory->read_surface(target).bytes == expected, "Callback changed captured transfer plan");
			std::cout << "Native image mapping publication:" << mapping_cases << " source/target metadata changes during real pending GPU work refused without VM publication;owned query survives timeout and retires on refusal/success;stable retry and95CPU-epoch checks,stale capture rejected,caller plan mutation cannot change encoding\n";
			std::cout << "Native scaled-image:" << cases << " GPU RGB565/ARGB8 nearest/linear/flip/clip/linear-Morton/padding cases;578rational boundaries/all65536RGB565 roundtrips;two CPU epochs,dirty source,target cache invalidation/retained old pixels,physical memmove,pending timeout/frozen source/padding survival,CAS refusal/retry,encode cancellation and no late teardown publication passed\n";
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
