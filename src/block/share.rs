impl ShareKey{
	/// create a share key from the inner data
	pub fn from_inner(generation:usize,lineage:u64,vtype:TypeId)->Self{
		Self{generation,lineage,vtype}
	}
	/// get the generation number
	pub fn get_generation(&self)->usize{self.generation}
	/// get the lineage id
	pub fn get_lineage(&self)->u64{self.lineage}
	/// gets the layer associated with this key if it exists
	pub fn get_shared<V:Any+Send>(&self)->Shared<V>{
		Shared{inner:Err(OnceCell::new()),key:*self}
	}
	/// get the type id
	pub fn get_type(&self)->TypeId{self.vtype}
	/// create a new generation of a new lineage with the type
	pub fn new<V:Any>()->Self{
		Self{
			generation:0,
			lineage:rand::random(),
			vtype:TypeId::of::<V>()
		}
	}
}
impl<'a,V:Any+Deserialize<'a>+Send> Deserialize<'a> for Shared<V>{
	fn deserialize<D:Deserializer<'a>>(deserializer:D)->Result<Self,D::Error>{
		let (inner,generation,lineage):(Option<Arc<Mutex<V>>>,usize,u64)=Deserialize::deserialize(deserializer)?;
		let inner=inner.ok_or_else(OnceCell::new);
		let key=ShareKey{generation,lineage,vtype:TypeId::of::<V>()};

		if let Ok(l)=&inner{
			SHARE_MAP.get_or_init(Default::default).insert(key,Box::new(Arc::downgrade(&l)));
		}
		Ok(Self{inner,key})
	}
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Shared<V>{
	fn valid(&self)->Self::InnerModule{self._derive(V::valid)}
	type InnerModule=Shared<W>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Shared<V>{
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self._do_layer(|layer|layer.embed(input,inputclasses,inputencoding))}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self._do_layer(|layer|layer.embed_mut(input,inputclasses,inputencoding))}
	fn forward(&self,input:Value<B>)->Value<B>{self._do_layer(|layer|layer.forward(input))}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self._do_layer(|layer|layer.forward_mut(input))}
	fn supports(&self,encoding:u64)->bool{self._do_layer(|layer|layer.supports(encoding))}
	type BlockWith<C:Backend>=Shared<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Shared<V>{
	fn collect_devices(&self,mut devices:Vec<B::Device>)->Vec<B::Device>{
		self._visit(|v|devices=v.collect_devices(mem::take(&mut devices)));
		devices
	}
	fn fork(self,device:&B::Device)->Self{self._map(|v|v.fork(device))}
	fn into_record(self)->Self::Record{
		let mut layer=None;
		let key=self.key;

		self._visit(|v|layer=Some(v.clone()));
		(layer,key.generation,key.lineage).into_record()
	}
	fn load_record(self,record:Self::Record)->Self{
		let mut layer=None;
		let mut key=self.key;

		self._visit(|v|layer=Some(v.clone()));

		(layer,key.generation,key.lineage)=(layer,key.generation,key.lineage).load_record(record);

		let inner=layer.ok_or_else(OnceCell::new).map(|l|{
			let inner=Arc::new(Mutex::new(l));
			let maphandle=SHARE_MAP.get_or_init(Default::default);

			maphandle.insert(key,Box::from(Arc::downgrade(&inner)));
			inner
		});
		Self{inner,key}
	}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{self._map(|v|v.map(mapper))}
	fn to_device(self,device:&B::Device)->Self{self._map(|v|v.to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self._visit(|v|v.visit(visitor))}
	type Record=<(Option<V>,usize,u64) as Module<B>>::Record;
}
impl<V:Any+ModuleDisplay+Send> ModuleDisplay for Shared<V>{}
impl<V:Any+ModuleDisplay+Send> ModuleDisplayDefault for Shared<V>{
	fn content(&self,content:Content)->Option<Content>{
		let mut c=None;
		self._visit(|l|c=l.content(content));

		c
	}
}
impl<V:Any+Send> Shared<V>{
	/// derive a model from a reference
	fn _derive<F:FnOnce(&V)->U,U:Any+Send>(&self,f:F)->Shared<U>{
		let key=ShareKey{
			generation:self.key.generation+1,
			lineage:self.key.lineage^RELINE.get(),
			vtype:TypeId::of::<U>()
		};
		let inner=match self.inner.clone(){
			Err(_)=>Err(OnceCell::new()),
			Ok(x)=>{
				let layer=x.lock().unwrap();
				let maphandle=SHARE_MAP.get_or_init(Default::default);
				let x=Arc::new(Mutex::new(f(&*layer)));

				maphandle.insert(key,Box::from(Arc::downgrade(&x)));
				Ok(x)
			}
		};

		Shared{inner,key}
	}
	fn _do_layer<F:FnOnce(&mut V)->Y,Y>(&self,f:F)->Y{
		let inner=self._upgrade_inner();
		let mut lock=inner.lock().unwrap();

		f(&mut *lock)
	}
	/// map the inner value if this is a primary share
	fn _map<F:FnOnce(V)->V>(mut self,f:F)->Self where V:Clone{
		if let Ok(inner)=&mut self.inner{
			let mut layer=inner.lock().unwrap();
			*layer=f(layer.clone());
		}
		self
	}
	fn _upgrade_inner(&self)->Arc<Mutex<V>>{
		let maphandle=SHARE_MAP.get_or_init(Default::default);
		let get_static=||{
			let layerhandle=maphandle.get(&self.key)?;
			let layerhandle:&Weak<Mutex<V>>=layerhandle.downcast_ref()?;
			Some(layerhandle.clone())
		};

		match &self.inner{
			Err(x)=>x.get().cloned().or_else(get_static).and_then(|x|x.upgrade()),
			Ok(x)=>Some(x.clone())
		}.expect("An existing primary share should still be living")
	}
	/// visit the inner value if this is a primary share
	fn _visit<F:FnOnce(&V)>(&self,f:F){
		if let Ok(inner)=&self.inner{
			let layer=inner.lock().unwrap();
			f(&*layer);
		}
	}
	/// make this share a primary share of its key. A previous primary share should still exist, otherwise the inner layer will have been dropped and the method will panic. For correct map/visit behavior, exactly one primary share should be included with each model, so this shouldn't be used unless a strong reference to the layer is needed outside, or the original primary is going to be dropped early. Primary share status is preserved when cloning
	pub fn into_primary(mut self)->Self{
		self.make_primary();
		self
	}
	/// check if this is a primary share
	pub fn is_primary(&self)->bool{self.inner.is_ok()}
	/// make this share a primary share of its key. A previous primary share should still exist, otherwise the inner layer will have been dropped and the method will panic. For correct map/visit behavior, exactly one primary share should be included with each model, so this shouldn't be used unless a strong reference to the layer is needed outside, or the original primary is going to be dropped early. Primary share status is preserved when cloning
	pub fn make_primary(&mut self){self.inner=Ok(self._upgrade_inner())}
	/// create a new share from the inner layer. The result will be a 'primary' Shared reference that delegates to the inner module for mapping and visiting purposes, and its shares will be secondary shares referencing the same layer with the same key. For module map/visit methods to work correctly, exactly one primary share should be present per key per model. Primary share status is preserved when cloning
	pub fn new(inner:V)->Self{
		let inner=Arc::new(Mutex::new(inner));
		let key=ShareKey::new::<V>();
		let maphandle=SHARE_MAP.get_or_init(Default::default);

		maphandle.insert(key,Box::from(Arc::downgrade(&inner)));
		Self{inner:Ok(inner.into()),key}
	}
	/// create another non primary share with the same key
	pub fn share(&self)->Self{
		Self{inner:Err(OnceCell::new()),key:self.key}
	}
	/// create another share with the same key, then swap it with self before returning, effectively taking the primary status of self and putting it in the returned value, leaving self non primary
	pub fn share_swap(&mut self)->Self{
		let mut s=self.share();

		mem::swap(&mut s,self);
		s
	}
}
impl<V:Any+Clone+Send> Clone for Shared<V>{
	fn clone(&self)->Self{self._derive(V::clone)}
}
impl<V:Serialize> Serialize for Shared<V>{
	fn serialize<S:Serializer>(&self,serializer:S)->Result<S::Ok,S::Error>{(self.inner.as_ref().cloned().ok(),self.key.generation+1,self.key.lineage).serialize(serializer)}
}
impl<V> Drop for Shared<V>{
	fn drop(&mut self){
		if let Ok(layerhandle)=&self.inner{
			if Arc::strong_count(layerhandle)==1{
				SHARE_MAP.get_or_init(Default::default).remove(&self.key);
			}
		}
	}
}

/// make a derivative model such that its shared layers have the same share pattern but with new lineages independent of the original. Might not work correctly if inner shares are cloned by means other than Clone, or if clone spawns threads and then clones different shares of the same key on different threads.
pub fn break_lineage<V:Clone+Send>(module:V)->V{
	let reset=RELINE.get()==0;
	if reset{RELINE.set(rand::random())}

	let result=module.clone();
	if reset{RELINE.set(0)}

	result
}

#[derive(Debug)]
/// wraps the inner layer to allow parameter sharing. Each Share is identified by a key composed of it's generation, lineage, and type, and shares with the same key reference the same layer. Creating a Share with new creates a new lineage, while cloning or otherwise deriving from an existing Share results in the same lineage with the generation incremented. Shared layers are dropped when the primary share of that key is dropped. Use share::break_lineage to create a clone of a model with new lineages for shared layers that are associated with each other but not with descendants of the original model. On modules mapping/visiting only the primary layer with the same share key will apply those to the inner, and on serialization only one layer with the same share key will serialize, so keep Shares of the same key in the same model.
pub struct Shared<V>{inner:Result<Arc<Mutex<V>>,OnceCell<Weak<Mutex<V>>>>,key:ShareKey}
#[derive(Clone,Copy,Debug,Eq,Hash,PartialEq)]
/// the key type for identifying shares
pub struct ShareKey{generation:usize,lineage:u64,vtype:TypeId}

/// map share keys to share info
static SHARE_MAP:OnceLock<DashMap<ShareKey,Box<dyn Any+Send+Sync>>>=OnceLock::new();
thread_local!{
	static RELINE:Cell<u64>=const{Cell::new(0)};
}

use burn::{
	module::{AutodiffModule,Content,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor},
	prelude::*,
	tensor::backend::AutodiffBackend
};
use dashmap::DashMap;
use serde::{Deserialize,Deserializer,Serialize,Serializer};
use super::{BlockVariant,Value};
use std::{
	any::{Any,TypeId},cell::{Cell,OnceCell},mem,sync::{Arc,Mutex,OnceLock,Weak}
};
